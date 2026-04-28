import time
from collections import deque
from typing import Optional


class MetricState:
    """
    Gerencia estado temporal de uma métrica com confiança suavizada.
    
    Armazena:
    - valor bruto atual
    - valor suavizado (EMA)
    - histórico curto para análise
    - histórico temporal em segundos
    - confiança entre 0.0 e 1.0
    - cooldown
    - tempo desde último evento válido
    - indicador de disponibilidade
    
    Atualização de confiança:
    confianca_nova = clip(
        confianca_antiga + ganho_subida * evidencia - ganho_descida * (1 - evidencia),
        0.0,
        1.0
    )
    """
    
    def __init__(
        self,
        name: str,
        gain_up: float = 0.15,
        gain_down: float = 0.05,
        ema_alpha: float = 0.3,
        history_size: int = 60,
        cooldown_seconds: float = 0.0
    ):
        """
        Args:
            name: Nome da métrica (ex: 'eye_closure', 'yawn', 'perclos')
            gain_up: Quanto a confiança sobe quando evidência é alta (padrão 0.15)
            gain_down: Quanto a confiança desce quando evidência é baixa (padrão 0.05)
            ema_alpha: Smoothing factor para EMA (0.0-1.0, padrão 0.3)
            history_size: Número máximo de valores no histórico (padrão 60)
            cooldown_seconds: Tempo mínimo entre eventos (padrão 0.0)
        """
        self.name = name
        self.gain_up = gain_up
        self.gain_down = gain_down
        self.ema_alpha = ema_alpha
        self.history_size = history_size
        self.cooldown_seconds = cooldown_seconds
        
        # Estado atual
        self.raw_value: float = 0.0
        self.smooth_value: float = 0.0
        self.confidence: float = 0.0
        self.is_available: bool = True
        
        # Histórico
        self.history: deque = deque(maxlen=history_size)
        self.timestamps: deque = deque(maxlen=history_size)
        
        # Timing
        self.last_event_time: float = -999.0
        self.last_update_time: float = time.time()
        self.cooldown_end_time: float = 0.0
    
    def update(self, raw_value: float, is_available: bool = True) -> None:
        """
        Atualiza o estado com novo valor bruto.
        
        Args:
            raw_value: Valor bruto da métrica (0.0-1.0 como evidência)
            is_available: Se a métrica está disponível (ex: face detectada)
        """
        current_time = time.time()
        self.last_update_time = current_time
        self.is_available = is_available
        
        if not is_available:
            # Se indisponível, degradar confiança lentamente
            self.confidence = max(0.0, self.confidence - self.gain_down)
            return
        
        # Armazenar valor bruto
        self.raw_value = max(0.0, min(1.0, raw_value))
        
        # Atualizar EMA (smoothing)
        if len(self.history) == 0:
            self.smooth_value = self.raw_value
        else:
            self.smooth_value = (
                self.ema_alpha * self.raw_value + 
                (1 - self.ema_alpha) * self.smooth_value
            )
        
        # Adicionar ao histórico
        self.history.append(self.smooth_value)
        self.timestamps.append(current_time)
        
        # Atualizar confiança com histerese
        # Evidência = quanto o valor suavizado indica que o evento está acontecendo
        evidence = self.smooth_value
        
        # Fórmula de atualização com ganhos assimétricos
        self.confidence += self.gain_up * evidence - self.gain_down * (1 - evidence)
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Registrar tempo do evento se confiança está alta
        if self.confidence > 0.7:
            self.last_event_time = current_time
    
    def is_on_cooldown(self) -> bool:
        """Retorna True se ainda está no período de cooldown."""
        if self.cooldown_seconds <= 0:
            return False
        return time.time() < self.cooldown_end_time
    
    def trigger_cooldown(self) -> None:
        """Inicia um período de cooldown."""
        self.cooldown_end_time = time.time() + self.cooldown_seconds
    
    def get_history_average(self, seconds: Optional[float] = None) -> float:
        """
        Retorna a média dos valores históricos nos últimos N segundos.
        Se seconds=None, usa todo o histórico.
        """
        if not self.history:
            return 0.0
        
        if seconds is None:
            return sum(self.history) / len(self.history)
        
        # Filtrar histórico pelos últimos N segundos
        current_time = time.time()
        cutoff_time = current_time - seconds
        
        recent_values = [
            val for val, ts in zip(self.history, self.timestamps)
            if ts >= cutoff_time
        ]
        
        if not recent_values:
            return 0.0
        
        return sum(recent_values) / len(recent_values)
    
    def time_since_event(self) -> float:
        """Retorna segundos desde o último evento de alta confiança."""
        return time.time() - self.last_event_time
    
    def reset(self) -> None:
        """Reseta todos os estados para valores iniciais."""
        self.raw_value = 0.0
        self.smooth_value = 0.0
        self.confidence = 0.0
        self.history.clear()
        self.timestamps.clear()
        self.last_event_time = -999.0
        self.cooldown_end_time = 0.0
    
    def get_debug_info(self) -> dict:
        """Retorna dicionário com informações de debug."""
        return {
            'name': self.name,
            'raw_value': round(self.raw_value, 3),
            'smooth_value': round(self.smooth_value, 3),
            'confidence': round(self.confidence, 3),
            'is_available': self.is_available,
            'on_cooldown': self.is_on_cooldown(),
            'time_since_event': round(self.time_since_event(), 2),
            'history_length': len(self.history),
        }
