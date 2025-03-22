# src/common/messaging.py
import json
import asyncio
import logging
from typing import Dict, Any, Callable, Optional, Union, List
import aio_pika
from aio_pika.abc import AbstractIncomingMessage
from tenacity import retry, stop_after_attempt, wait_exponential


class MessageBroker:
    """
    Client per la comunicazione asincrona tramite RabbitMQ.
    
    Implementa pattern di comunicazione pub/sub e RPC utilizzando
    RabbitMQ come message broker.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        vhost: str = "/",
        ssl: bool = False,
        connection_name: str = "predictsense-connection"
    ):
        """
        Inizializza il client RabbitMQ.
        
        Args:
            host: Host del server RabbitMQ
            port: Porta del server RabbitMQ
            username: Username per autenticazione
            password: Password per autenticazione
            vhost: Virtual host RabbitMQ
            ssl: Abilitare SSL/TLS
            connection_name: Nome della connessione per debugging
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.vhost = vhost
        self.ssl = ssl
        self.connection_name = connection_name
        
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.logger = logging.getLogger(__name__)
        
        # Traccia delle code e degli exchange attualmente dichiarati
        self._declared_exchanges: Dict[str, aio_pika.Exchange] = {}
        self._declared_queues: Dict[str, aio_pika.Queue] = {}
        
        # Traccia dei consumer attivi
        self._active_consumers: List[asyncio.Task] = []
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def connect(self):
        """Stabilisce la connessione con RabbitMQ."""
        if self.connection and not self.connection.is_closed:
            return
        
        # Costruisci connection string
        connection_string = f"amqp://{self.username}:{self.password}@{self.host}:{self.port}/{self.vhost}"
        
        self.logger.info(f"Connecting to RabbitMQ at {self.host}:{self.port}")
        
        # Tenta la connessione con retry automatico
        try:
            self.connection = await aio_pika.connect_robust(
                connection_string,
                client_properties={
                    "connection_name": self.connection_name
                }
            )
            
            # Configurazione del canale
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=10)
            
            # Configurazione degli handler per la chiusura della connessione
            self.connection.add_close_callback(self._on_connection_closed)
            
            self.logger.info("Successfully connected to RabbitMQ")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            raise
    
    async def close(self):
        """Chiude la connessione con RabbitMQ."""
        self.logger.info("Closing RabbitMQ connection")
        
        # Cancella tutti i consumer attivi
        for task in self._active_consumers:
            task.cancel()
        
        # Chiudi canale e connessione
        if self.channel and not self.channel.is_closed:
            await self.channel.close()
        
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
        
        self.logger.info("RabbitMQ connection closed")
    
    def _on_connection_closed(self, sender, exc):
        """Handler per gestire la chiusura della connessione."""
        self.logger.warning(f"RabbitMQ connection closed: {exc}")
    
    async def get_exchange(
        self,
        exchange_name: str,
        exchange_type: str = "topic",
        durable: bool = True,
        auto_delete: bool = False
    ) -> aio_pika.Exchange:
        """
        Ottiene un exchange, dichiarandolo se necessario.
        
        Args:
            exchange_name: Nome dell'exchange
            exchange_type: Tipo di exchange (direct, fanout, topic, headers)
            durable: Se l'exchange deve persistere dopo riavvii del broker
            auto_delete: Se l'exchange deve essere eliminato quando non ha più binding
        
        Returns:
            L'oggetto Exchange
        """
        if not self.channel:
            await self.connect()
        
        # Usa exchange già dichiarato se presente
        if exchange_name in self._declared_exchanges:
            return self._declared_exchanges[exchange_name]
        
        # Mappa il tipo di exchange
        exchange_type_map = {
            "direct": aio_pika.ExchangeType.DIRECT,
            "fanout": aio_pika.ExchangeType.FANOUT,
            "topic": aio_pika.ExchangeType.TOPIC,
            "headers": aio_pika.ExchangeType.HEADERS
        }
        
        exchange_type_enum = exchange_type_map.get(
            exchange_type, 
            aio_pika.ExchangeType.TOPIC
        )
        
        # Dichiara l'exchange
        exchange = await self.channel.declare_exchange(
            name=exchange_name,
            type=exchange_type_enum,
            durable=durable,
            auto_delete=auto_delete
        )
        
        # Salva per riuso
        self._declared_exchanges[exchange_name] = exchange
        
        return exchange
    
    async def get_queue(
        self,
        queue_name: str,
        durable: bool = True,
        exclusive: bool = False,
        auto_delete: bool = False,
        arguments: Optional[Dict[str, Any]] = None
    ) -> aio_pika.Queue:
        """
        Ottiene una coda, dichiarandola se necessario.
        
        Args:
            queue_name: Nome della coda
            durable: Se la coda deve persistere dopo riavvii del broker
            exclusive: Se la coda può essere utilizzata solo dalla connessione corrente
            auto_delete: Se la coda deve essere eliminata quando non ha più consumer
            arguments: Argomenti aggiuntivi per la coda
        
        Returns:
            L'oggetto Queue
        """
        if not self.channel:
            await self.connect()
        
        # Usa coda già dichiarata se presente
        if queue_name in self._declared_queues:
            return self._declared_queues[queue_name]
        
        # Dichiara la coda
        queue = await self.channel.declare_queue(
            name=queue_name,
            durable=durable,
            exclusive=exclusive,
            auto_delete=auto_delete,
            arguments=arguments or {}
        )
        
        # Salva per riuso
        self._declared_queues[queue_name] = queue
        
        return queue
    
    async def publish(
        self,
        exchange_name: str,
        routing_key: str,
        message: Union[Dict[str, Any], str],
        headers: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        expiration: Optional[int] = None,
        priority: Optional[int] = None,
        delivery_mode: int = 2  # 2 = persistente
    ):
        """
        Pubblica un messaggio su un exchange.
        
        Args:
            exchange_name: Nome dell'exchange
            routing_key: Routing key per il messaggio
            message: Messaggio da pubblicare (dict o str)
            headers: Header del messaggio
            correlation_id: ID di correlazione per pattern RPC
            reply_to: Coda di risposta per pattern RPC
            expiration: TTL del messaggio in ms
            priority: Priorità del messaggio (0-9)
            delivery_mode: Modalità di delivery (1=non persistente, 2=persistente)
        """
        if not self.channel:
            await self.connect()
        
        # Ottieni o dichiara l'exchange
        exchange = await self.get_exchange(exchange_name)
        
        # Prepara il messaggio
        if isinstance(message, dict):
            message_body = json.dumps(message).encode()
        elif isinstance(message, str):
            message_body = message.encode()
        else:
            message_body = str(message).encode()
        
        # Costruisci le proprietà del messaggio
        message_properties = {}
        
        if headers:
            message_properties["headers"] = headers
        
        if correlation_id:
            message_properties["correlation_id"] = correlation_id
        
        if reply_to:
            message_properties["reply_to"] = reply_to
        
        if expiration:
            message_properties["expiration"] = str(expiration)
        
        if priority is not None:
            message_properties["priority"] = priority
        
        message_properties["delivery_mode"] = delivery_mode
        
        # Crea il messaggio
        amqp_message = aio_pika.Message(
            body=message_body,
            **message_properties
        )
        
        # Pubblica il messaggio
        await exchange.publish(
            message=amqp_message,
            routing_key=routing_key
        )
        
        self.logger.debug(
            f"Published message to exchange '{exchange_name}' "
            f"with routing key '{routing_key}'"
        )
    
    async def subscribe(
        self,
        queue_name: str,
        callback: Callable[[Dict[str, Any], AbstractIncomingMessage], Any],
        exchange_name: Optional[str] = None,
        routing_key: str = "#",
        auto_ack: bool = False
    ):
        """
        Sottoscrive a una coda per consumare messaggi.
        
        Args:
            queue_name: Nome della coda
            callback: Funzione di callback da chiamare per ogni messaggio
            exchange_name: Nome dell'exchange (opzionale)
            routing_key: Routing key per binding (se exchange specificato)
            auto_ack: Se i messaggi devono essere ack automaticamente
        """
        if not self.channel:
            await self.connect()
        
        # Ottieni o dichiara la coda
        queue = await self.get_queue(queue_name)
        
        # Se specificato un exchange, crea un binding
        if exchange_name:
            exchange = await self.get_exchange(exchange_name)
            await queue.bind(exchange, routing_key)
        
        # Wrapper per il callback che gestisce la deserializzazione
        async def message_handler(message: AbstractIncomingMessage):
            async with message.process(auto_ack=auto_ack):
                # Deserializza il messaggio
                try:
                    message_body = message.body.decode()
                    payload = json.loads(message_body)
                except json.JSONDecodeError:
                    payload = message.body.decode()
                except UnicodeDecodeError:
                    payload = message.body
                
                # Invoca il callback
                await callback(payload, message)
        
        # Inizia a consumare messaggi
        consumer_tag = await queue.consume(message_handler)
        
        self.logger.info(
            f"Subscribed to queue '{queue_name}'"
            + (f" bound to exchange '{exchange_name}'" if exchange_name else "")
        )
        
        # Tieni traccia del consumer
        consumer_task = asyncio.create_task(
            self._keep_consumer_alive(queue_name, callback, exchange_name, routing_key, auto_ack)
        )
        self._active_consumers.append(consumer_task)
        
        return consumer_tag
    
    async def _keep_consumer_alive(
        self,
        queue_name: str,
        callback: Callable,
        exchange_name: Optional[str],
        routing_key: str,
        auto_ack: bool
    ):
        """Mantiene attivo il consumer anche in caso di disconnessione."""
        while True:
            try:
                await asyncio.sleep(60)  # Check periodico
            except asyncio.CancelledError:
                self.logger.info(f"Consumer for queue '{queue_name}' cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in consumer keep-alive: {str(e)}")
                
                # Se la connessione è stata persa, riconnetti e ricrea il consumer
                if not self.connection or self.connection.is_closed:
                    try:
                        self.logger.info(f"Reconnecting consumer for queue '{queue_name}'")
                        await self.connect()
                        await self.subscribe(queue_name, callback, exchange_name, routing_key, auto_ack)
                        break  # Esci dal loop, il nuovo consumer ha il suo task
                    except Exception as conn_err:
                        self.logger.error(f"Failed to reconnect consumer: {str(conn_err)}")
                        await asyncio.sleep(5)  # Attendi prima di riprovare