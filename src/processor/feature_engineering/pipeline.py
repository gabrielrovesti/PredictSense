"""
Definisce l'interfaccia base per le pipeline di feature engineering.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class Pipeline(ABC):
    """
    Interfaccia base per le pipeline di feature engineering.
    
    Definisce i metodi che ogni pipeline deve implementare per
    trasformare i dati grezzi in features utilizzabili dai modelli.
    """
    
    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa i dati applicando le trasformazioni.
        
        Args:
            data: Dati grezzi da processare
        
        Returns:
            Dati processati con features estratte
        """
        pass
    
    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trasforma i dati senza stato interno.
        
        Utile per applicare le stesse trasformazioni a nuovi dati
        senza effetti collaterali.
        
        Args:
            data: Dati grezzi da trasformare
        
        Returns:
            Dati trasformati
        """
        pass