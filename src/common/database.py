import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from contextlib import asynccontextmanager

import asyncpg
from tenacity import retry, stop_after_attempt, wait_exponential


class Database:
    """
    Client per interagire con il database PostgreSQL.
    
    Fornisce connessione pooled e metodi per operazioni comuni sul database.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        username: str = "postgres",
        password: str = "postgres",
        database: str = "predictsense",
        max_connections: int = 10,
        min_connections: int = 2,
        connection_timeout: float = 60.0,
        ssl: bool = False
    ):
        """
        Inizializza il client database.
        
        Args:
            host: Hostname del server PostgreSQL
            port: Porta del server PostgreSQL
            username: Username per autenticazione
            password: Password per autenticazione
            database: Nome del database
            max_connections: Numero massimo di connessioni nel pool
            min_connections: Numero minimo di connessioni nel pool
            connection_timeout: Timeout per le connessioni in secondi
            ssl: Abilita SSL/TLS per la connessione
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.connection_timeout = connection_timeout
        self.ssl = ssl
        
        self.pool = None
        self.logger = logging.getLogger(__name__)
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def connect(self):
        """Stabilisce la connessione al database creando un pool di connessioni."""
        if self.pool is not None:
            return
        
        try:
            self.logger.info(f"Connecting to PostgreSQL at {self.host}:{self.port}/{self.database}")
            
            # Crea il pool di connessioni
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                database=self.database,
                min_size=self.min_connections,
                max_size=self.max_connections,
                timeout=self.connection_timeout,
                command_timeout=self.connection_timeout,
                ssl=self.ssl
            )
            
            self.logger.info("Connected to PostgreSQL successfully")
            
            # Verifica connessione
            async with self.pool.acquire() as connection:
                version = await connection.fetchval("SELECT version()")
                self.logger.info(f"PostgreSQL version: {version}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            raise
    
    async def close(self):
        """Chiude tutte le connessioni nel pool."""
        if self.pool is not None:
            self.logger.info("Closing PostgreSQL connection pool")
            await self.pool.close()
            self.pool = None
            self.logger.info("PostgreSQL connection pool closed")
    
    async def is_connected(self) -> bool:
        """Verifica se il database è connesso."""
        if self.pool is None:
            return False
        
        try:
            async with self.pool.acquire() as connection:
                await connection.fetchval("SELECT 1")
            return True
        except Exception:
            return False
    
    @asynccontextmanager
    async def transaction(self):
        """
        Context manager per le transazioni.
        
        Esempio:
        ```
        async with db.transaction() as conn:
            await conn.execute("INSERT INTO ...")
            await conn.execute("UPDATE ...")
        ```
        """
        if self.pool is None:
            await self.connect()
        
        async with self.pool.acquire() as connection:
            transaction = connection.transaction()
            await transaction.start()
            
            try:
                yield connection
                await transaction.commit()
            except Exception:
                await transaction.rollback()
                raise
    
    async def execute(self, query: str, *args) -> str:
        """
        Esegue una query e restituisce lo stato.
        
        Args:
            query: Query SQL
            *args: Parametri per la query
        
        Returns:
            Stato dell'esecuzione
        """
        if self.pool is None:
            await self.connect()
        
        async with self.pool.acquire() as connection:
            return await connection.execute(query, *args)
    
    async def fetch_all(self, query: str, *args) -> List[asyncpg.Record]:
        """
        Esegue una query e restituisce tutte le righe.
        
        Args:
            query: Query SQL
            *args: Parametri per la query
        
        Returns:
            Lista di record
        """
        if self.pool is None:
            await self.connect()
        
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
    
    async def fetch_one(self, query: str, *args) -> Optional[asyncpg.Record]:
        """
        Esegue una query e restituisce la prima riga.
        
        Args:
            query: Query SQL
            *args: Parametri per la query
        
        Returns:
            Record se trovato, None altrimenti
        """
        if self.pool is None:
            await self.connect()
        
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *args)
    
    async def fetch_val(self, query: str, *args) -> Any:
        """
        Esegue una query e restituisce un singolo valore.
        
        Args:
            query: Query SQL
            *args: Parametri per la query
        
        Returns:
            Valore se trovato, None altrimenti
        """
        if self.pool is None:
            await self.connect()
        
        async with self.pool.acquire() as connection:
            return await connection.fetchval(query, *args)
    
    async def execute_many(self, query: str, args: List[Tuple]):
        """
        Esegue una query più volte con parametri diversi.
        
        Args:
            query: Query SQL
            args: Lista di tuple di parametri
        """
        if self.pool is None:
            await self.connect()
        
        async with self.pool.acquire() as connection:
            await connection.executemany(query, args)
    
    async def execute_batch(self, queries: List[str]):
        """
        Esegue un batch di query in una singola transazione.
        
        Args:
            queries: Lista di query SQL
        """
        async with self.transaction() as connection:
            for query in queries:
                await connection.execute(query)
    
    async def create_tables_from_file(self, file_path: str):
        """
        Crea tabelle da un file SQL.
        
        Args:
            file_path: Percorso del file SQL
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"SQL file not found: {file_path}")
        
        with open(file_path, 'r') as sql_file:
            sql = sql_file.read()
        
        if self.pool is None:
            await self.connect()
        
        async with self.pool.acquire() as connection:
            await connection.execute(sql)
    
    async def table_exists(self, table_name: str) -> bool:
        """
        Verifica se una tabella esiste.
        
        Args:
            table_name: Nome della tabella
        
        Returns:
            True se la tabella esiste, False altrimenti
        """
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public'
                AND table_name = $1
            )
        """
        
        return await self.fetch_val(query, table_name)
    
    async def get_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Ottiene informazioni sulle colonne di una tabella.
        
        Args:
            table_name: Nome della tabella
        
        Returns:
            Lista di dizionari con informazioni sulle colonne
        """
        query = """
            SELECT 
                column_name, 
                data_type,
                is_nullable,
                column_default
            FROM 
                information_schema.columns
            WHERE 
                table_schema = 'public'
                AND table_name = $1
            ORDER BY 
                ordinal_position
        """
        
        columns = await self.fetch_all(query, table_name)
        return [dict(col) for col in columns]


