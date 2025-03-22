import uuid
import asyncio
import logging
from typing import Dict, Any, Optional, Union, Callable
from .messaging import MessageBroker


class RPCClient:
    """
    Implementazione del pattern RPC (Remote Procedure Call) con RabbitMQ.
    
    Permette di inviare richieste e attendere risposte in modo asincrono.
    """
    
    def __init__(self, message_broker: MessageBroker):
        """
        Inizializza il client RPC.
        
        Args:
            message_broker: Istanza di MessageBroker
        """
        self.message_broker = message_broker
        self.logger = logging.getLogger(__name__)
        
        # Dizionario per tenere traccia delle richieste in attesa
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
        # Flag per indicare se il consumer è attivo
        self.consumer_active = False
        self.callback_queue_name = f"rpc.response.{uuid.uuid4()}"
    
    async def initialize(self):
        """Inizializza il client RPC."""
        # Crea una coda di callback esclusiva per le risposte
        await self.message_broker.get_queue(
            queue_name=self.callback_queue_name,
            exclusive=True,
            auto_delete=True
        )
        
        # Sottoscrivi alla coda di callback per ricevere le risposte
        await self.message_broker.subscribe(
            queue_name=self.callback_queue_name,
            callback=self._on_response,
            auto_ack=True
        )
        
        self.consumer_active = True
        self.logger.info(f"RPC client initialized with callback queue: {self.callback_queue_name}")
    
    async def _on_response(self, payload, message):
        """Callback invocato quando arriva una risposta."""
        correlation_id = message.correlation_id
        
        if correlation_id in self.pending_requests:
            future = self.pending_requests.pop(correlation_id)
            if not future.done():
                future.set_result(payload)
        else:
            self.logger.warning(f"Received response with unknown correlation_id: {correlation_id}")
    
    async def call(
        self,
        exchange_name: str,
        routing_key: str,
        payload: Union[Dict[str, Any], str],
        timeout: float = 30.0,
        headers: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Esegue una chiamata RPC.
        
        Args:
            exchange_name: Nome dell'exchange
            routing_key: Routing key per la richiesta
            payload: Payload della richiesta
            timeout: Timeout in secondi
            headers: Header opzionali
        
        Returns:
            La risposta ricevuta
        
        Raises:
            asyncio.TimeoutError: Se non arriva risposta entro il timeout
        """
        if not self.consumer_active:
            await self.initialize()
        
        # Genera correlation ID per tracciare la richiesta
        correlation_id = str(uuid.uuid4())
        
        # Crea future per attendere la risposta
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[correlation_id] = future
        
        # Pubblica la richiesta
        await self.message_broker.publish(
            exchange_name=exchange_name,
            routing_key=routing_key,
            message=payload,
            headers=headers,
            correlation_id=correlation_id,
            reply_to=self.callback_queue_name
        )
        
        self.logger.debug(
            f"Sent RPC request to '{exchange_name}:{routing_key}' "
            f"with correlation_id: {correlation_id}"
        )
        
        try:
            # Attendi la risposta con timeout
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            # Rimuovi la richiesta pendente
            self.pending_requests.pop(correlation_id, None)
            self.logger.warning(
                f"RPC request to '{exchange_name}:{routing_key}' "
                f"with correlation_id: {correlation_id} timed out after {timeout}s"
            )
            raise
        except Exception as e:
            # Rimuovi la richiesta pendente
            self.pending_requests.pop(correlation_id, None)
            self.logger.error(f"Error in RPC call: {str(e)}")
            raise


class EventEmitter:
    """
    Utility per pubblicare e sottoscrivere a eventi di sistema.
    
    Implementa pattern pub/sub tramite RabbitMQ.
    """
    
    def __init__(
        self,
        message_broker: MessageBroker,
        exchange_name: str = "events"
    ):
        """
        Inizializza l'emitter di eventi.
        
        Args:
            message_broker: Istanza di MessageBroker
            exchange_name: Nome dell'exchange per gli eventi
        """
        self.message_broker = message_broker
        self.exchange_name = exchange_name
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Inizializza l'emitter dichiarando l'exchange."""
        await self.message_broker.get_exchange(
            exchange_name=self.exchange_name,
            exchange_type="topic",
            durable=True
        )
        self.logger.info(f"Event emitter initialized with exchange: {self.exchange_name}")
    
    async def emit(
        self,
        event_type: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, Any]] = None
    ):
        """
        Emette un evento.
        
        Args:
            event_type: Tipo di evento (usato come routing key)
            payload: Payload dell'evento
            headers: Header opzionali
        """
        # Costruisci il messaggio con metadati
        event_message = {
            "event_type": event_type,
            "timestamp": self._get_timestamp(),
            "payload": payload
        }
        
        # Pubblica l'evento
        await self.message_broker.publish(
            exchange_name=self.exchange_name,
            routing_key=event_type,
            message=event_message,
            headers=headers
        )
        
        self.logger.debug(f"Emitted event of type: {event_type}")
    
    async def on(
        self,
        event_type: str,
        callback: Callable,
        queue_name: Optional[str] = None
    ):
        """
        Sottoscrive a un tipo di evento.
        
        Args:
            event_type: Tipo di evento (routing key)
            callback: Funzione di callback
            queue_name: Nome della coda (generato automaticamente se None)
        """
        # Se queue_name non specificato, genera un nome
        if queue_name is None:
            queue_name = f"events.{event_type}.{uuid.uuid4()}"
        
        # Sottoscrivi all'evento
        await self.message_broker.subscribe(
            queue_name=queue_name,
            callback=callback,
            exchange_name=self.exchange_name,
            routing_key=event_type,
            auto_ack=False
        )
        
        self.logger.info(f"Subscribed to event type: {event_type} with queue: {queue_name}")
        
        return queue_name
    
    def _get_timestamp(self) -> str:
        """Restituisce il timestamp corrente in formato ISO."""
        from datetime import datetime
        return datetime.utcnow().isoformat()