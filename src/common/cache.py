import json
import logging
from typing import Any, Dict, List, Optional, Set, Union

import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential


class CacheClient:
    """
    Client per interagire con Redis come cache.
    
    Fornisce metodi per operazioni comuni di caching.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 10.0,
        max_connections: int = 10
    ):
        """
        Inizializza il client Redis.
        
        Args:
            host: Hostname del server Redis
            port: Porta del server Redis
            password: Password per autenticazione
            db: Indice del database Redis
            socket_timeout: Timeout per le operazioni in secondi
            socket_connect_timeout: Timeout per la connessione in secondi
            max_connections: Numero massimo di connessioni nel pool
        """
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.max_connections = max_connections
        
        self.client = None
        self.logger = logging.getLogger(__name__)
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def connect(self):
        """Stabilisce la connessione a Redis."""
        if self.client is not None:
            return
        
        try:
            self.logger.info(f"Connecting to Redis at {self.host}:{self.port}/{self.db}")
            
            # Crea il client Redis
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                max_connections=self.max_connections,
                decode_responses=True  # Auto-decode response from bytes to str
            )
            
            # Verifica connessione
            await self.client.ping()
            
            self.logger.info("Connected to Redis successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
    
    async def close(self):
        """Chiude la connessione a Redis."""
        if self.client is not None:
            self.logger.info("Closing Redis connection")
            await self.client.aclose()
            self.client = None
            self.logger.info("Redis connection closed")
    
    async def is_connected(self) -> bool:
        """Verifica se Redis è connesso."""
        if self.client is None:
            return False
        
        try:
            return await self.client.ping()
        except Exception:
            return False
    
    async def get(self, key: str) -> Any:
        """
        Recupera un valore dalla cache.
        
        Args:
            key: Chiave del valore
        
        Returns:
            Valore se trovato, None altrimenti
        """
        if self.client is None:
            await self.connect()
        
        value = await self.client.get(key)
        return value
    
    async def set(
        self, 
        key: str, 
        value: str, 
        expire: Optional[int] = None
    ) -> bool:
        """
        Imposta un valore nella cache.
        
        Args:
            key: Chiave del valore
            value: Valore da salvare
            expire: TTL in secondi (opzionale)
        
        Returns:
            True se l'operazione ha successo, False altrimenti
        """
        if self.client is None:
            await self.connect()
        
        try:
            if expire is not None:
                return await self.client.setex(key, expire, value)
            else:
                return await self.client.set(key, value)
        except Exception as e:
            self.logger.error(f"Error setting cache key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> int:
        """
        Elimina una chiave dalla cache.
        
        Args:
            key: Chiave da eliminare
        
        Returns:
            Numero di chiavi eliminate
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.delete(key)
    
    async def exists(self, key: str) -> bool:
        """
        Verifica se una chiave esiste nella cache.
        
        Args:
            key: Chiave da verificare
        
        Returns:
            True se la chiave esiste, False altrimenti
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.exists(key) > 0
    
    async def expire(self, key: str, seconds: int) -> bool:
        """
        Imposta il TTL per una chiave.
        
        Args:
            key: Chiave da modificare
            seconds: TTL in secondi
        
        Returns:
            True se l'operazione ha successo, False altrimenti
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.expire(key, seconds)
    
    async def ttl(self, key: str) -> int:
        """
        Ottiene il TTL rimanente per una chiave.
        
        Args:
            key: Chiave da verificare
        
        Returns:
            TTL in secondi, -1 se la chiave non ha TTL, -2 se la chiave non esiste
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.ttl(key)
    
    async def keys(self, pattern: str) -> List[str]:
        """
        Trova chiavi che corrispondono a un pattern.
        
        Args:
            pattern: Pattern di ricerca (es. "user:*")
        
        Returns:
            Lista di chiavi corrispondenti
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.keys(pattern)
    
    async def flush_db(self) -> bool:
        """
        Elimina tutti i dati dal database attuale.
        
        Returns:
            True se l'operazione ha successo
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.flushdb()
    
    # Hash operations
    
    async def hget(self, name: str, key: str) -> str:
        """
        Recupera un valore da un hash.
        
        Args:
            name: Nome dell'hash
            key: Chiave nel hash
        
        Returns:
            Valore se trovato, None altrimenti
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.hget(name, key)
    
    async def hset(self, name: str, key: str, value: str) -> int:
        """
        Imposta un valore in un hash.
        
        Args:
            name: Nome dell'hash
            key: Chiave nel hash
            value: Valore da salvare
        
        Returns:
            1 se è stato creato un nuovo campo, 0 se è stato aggiornato
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.hset(name, key, value)
    
    async def hmset(self, name: str, mapping: Dict[str, str]) -> bool:
        """
        Imposta più valori in un hash.
        
        Args:
            name: Nome dell'hash
            mapping: Dizionario di chiavi e valori
        
        Returns:
            True se l'operazione ha successo
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.hmset(name, mapping)
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """
        Recupera tutti i valori da un hash.
        
        Args:
            name: Nome dell'hash
        
        Returns:
            Dizionario di chiavi e valori
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.hgetall(name)
    
    async def hdel(self, name: str, *keys) -> int:
        """
        Elimina una o più chiavi da un hash.
        
        Args:
            name: Nome dell'hash
            *keys: Chiavi da eliminare
        
        Returns:
            Numero di chiavi eliminate
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.hdel(name, *keys)
    
    # List operations
    
    async def lpush(self, name: str, *values) -> int:
        """
        Aggiunge valori all'inizio di una lista.
        
        Args:
            name: Nome della lista
            *values: Valori da aggiungere
        
        Returns:
            Lunghezza della lista dopo l'operazione
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.lpush(name, *values)
    
    async def rpush(self, name: str, *values) -> int:
        """
        Aggiunge valori alla fine di una lista.
        
        Args:
            name: Nome della lista
            *values: Valori da aggiungere
        
        Returns:
            Lunghezza della lista dopo l'operazione
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.rpush(name, *values)
    
    async def lpop(self, name: str) -> str:
        """
        Rimuove e restituisce il primo elemento di una lista.
        
        Args:
            name: Nome della lista
        
        Returns:
            Valore rimosso, None se la lista è vuota
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.lpop(name)
    
    async def rpop(self, name: str) -> str:
        """
        Rimuove e restituisce l'ultimo elemento di una lista.
        
        Args:
            name: Nome della lista
        
        Returns:
            Valore rimosso, None se la lista è vuota
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.rpop(name)
    
    async def lrange(self, name: str, start: int, end: int) -> List[str]:
        """
        Recupera un intervallo di elementi da una lista.
        
        Args:
            name: Nome della lista
            start: Indice iniziale
            end: Indice finale
        
        Returns:
            Lista di elementi
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.lrange(name, start, end)
    
    # Set operations
    
    async def sadd(self, name: str, *values) -> int:
        """
        Aggiunge valori a un set.
        
        Args:
            name: Nome del set
            *values: Valori da aggiungere
        
        Returns:
            Numero di valori aggiunti
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.sadd(name, *values)
    
    async def smembers(self, name: str) -> Set[str]:
        """
        Recupera tutti i membri di un set.
        
        Args:
            name: Nome del set
        
        Returns:
            Set di membri
        """
        if self.client is None:
            await self.connect()
        
        return await self.client.smembers(name)
    
    # Utility methods
    
    async def cache_json(
        self, 
        key: str, 
        data: Any, 
        expire: Optional[int] = None
    ) -> bool:
        """
        Serializza e memorizza dati JSON nella cache.
        
        Args:
            key: Chiave del valore
            data: Dati da serializzare
            expire: TTL in secondi (opzionale)
        
        Returns:
            True se l'operazione ha successo, False altrimenti
        """
        try:
            json_data = json.dumps(data)
            return await self.set(key, json_data, expire)
        except Exception as e:
            self.logger.error(f"Error caching JSON for key {key}: {str(e)}")
            return False
    
    async def get_json(self, key: str) -> Any:
        """
        Recupera e deserializza dati JSON dalla cache.
        
        Args:
            key: Chiave del valore
        
        Returns:
            Dati deserializzati se trovati, None altrimenti
        """
        data = await self.get(key)
        if data is None:
            return None
        
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON for key {key}: {str(e)}")
            return None
    
    async def get_or_set(
        self, 
        key: str, 
        callback: callable, 
        expire: Optional[int] = None
    ) -> Any:
        """
        Recupera un valore dalla cache o lo imposta se non esiste.
        
        Args:
            key: Chiave del valore
            callback: Funzione asincrona da chiamare per ottenere il valore
            expire: TTL in secondi (opzionale)
        
        Returns:
            Valore dalla cache o dal callback
        """
        value = await self.get(key)
        
        if value is None:
            value = await callback()
            if value is not None:
                await self.set(key, value, expire)
        
        return value
    
    async def get_or_set_json(
        self, 
        key: str, 
        callback: callable, 
        expire: Optional[int] = None
    ) -> Any:
        """
        Recupera un valore JSON dalla cache o lo imposta se non esiste.
        
        Args:
            key: Chiave del valore
            callback: Funzione asincrona da chiamare per ottenere il valore
            expire: TTL in secondi (opzionale)
        
        Returns:
            Valore deserializzato dalla cache o dal callback
        """
        value = await self.get_json(key)
        
        if value is None:
            value = await callback()
            if value is not None:
                await self.cache_json(key, value, expire)
        
        return value