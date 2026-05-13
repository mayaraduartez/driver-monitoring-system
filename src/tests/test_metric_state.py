"""
Teste prático do MetricState
Simula sensores reais e mostra evolução da confiança
"""

import sys
sys.path.insert(0, '/Users/mayaraduarte/Documents/tcc/landmarkAI/src')

from utils.metric_state import MetricState
import time


def test_1_basico():
    """Teste 1: Comportamento básico com valores constantes"""
    print("\n" + "="*60)
    print("TESTE 1: Olho FECHADO por 10 frames")
    print("="*60)
    
    eye_state = MetricState('eye_closure', gain_up=0.15, gain_down=0.05, ema_alpha=0.3)
    
    # Simular olho fechado (EAR baixo = 0.08)
    for frame in range(10):
        raw_ear = 0.08
        evidencia = 1.0 - raw_ear  # Baixo EAR = alta evidência
        
        eye_state.update(evidencia)
        
        print(f"Frame {frame+1:2d}: raw={raw_ear:.2f} | smooth={eye_state.smooth_value:.3f} | confidence={eye_state.confidence:.3f}")
    
    print(f"\n✅ Resultado: confidence={eye_state.confidence:.3f}")
    print(f"   → Alerta? {eye_state.confidence > 0.7}")


def test_2_com_ruido():
    """Teste 2: Olho fechado mas COM ruído (piscada falsa)"""
    print("\n" + "="*60)
    print("TESTE 2: Olho FECHADO mas COM RUÍDO (piscada no meio)")
    print("="*60)
    
    eye_state = MetricState('eye_closure', gain_up=0.15, gain_down=0.05, ema_alpha=0.3)
    
    # Sequência realista
    valores = [
        0.08,  # Fechando
        0.07,  # Fechado
        0.09,  # Fechado
        0.92,  # 💥 RUÍDO! Piscou
        0.06,  # De novo fechado
        0.07,  # Fechado
        0.08,  # Fechado
        0.07,  # Fechado
    ]
    
    for frame, raw_ear in enumerate(valores):
        evidencia = 1.0 - raw_ear
        eye_state.update(evidencia)
        
        print(f"Frame {frame+1}: raw={raw_ear:.2f} | smooth={eye_state.smooth_value:.3f} | confidence={eye_state.confidence:.3f}")
    
    print(f"\n✅ Resultado: confidence={eye_state.confidence:.3f}")
    print(f"   → Note: Ruído em frame 4 quase não afeta (EMA suaviza)")


def test_3_transicao():
    """Teste 3: Transição de ABERTO → FECHADO"""
    print("\n" + "="*60)
    print("TESTE 3: Olho ABRE → FECHA gradualmente")
    print("="*60)
    
    eye_state = MetricState('eye_closure', gain_up=0.15, gain_down=0.05)
    
    # Progression realista
    valores = [
        0.25,  # Aberto
        0.25,  # Aberto
        0.20,  # Começando a fechar
        0.15,  # Fechando
        0.10,  # Mais fechado
        0.08,  # Bem fechado
        0.07,  # Bem fechado
        0.06,  # Bem fechado
    ]
    
    for frame, raw_ear in enumerate(valores):
        evidencia = 1.0 - raw_ear
        eye_state.update(evidencia)
        
        status = "🟢 NORMAL" if eye_state.confidence < 0.3 else "🟡 ATENÇÃO" if eye_state.confidence < 0.7 else "🔴 ALERTA"
        print(f"Frame {frame+1}: raw={raw_ear:.2f} | confidence={eye_state.confidence:.3f} {status}")
    
    print(f"\n✅ Ponto de ALERTA atingido em frame ~7")


def test_4_cooldown():
    """Teste 4: Cooldown bloqueando alertas repetidos"""
    print("\n" + "="*60)
    print("TESTE 4: COOLDOWN (evita 30 alertas por segundo)")
    print("="*60)
    
    eye_state = MetricState('eye_closure', cooldown_seconds=2.0)  # 2 segundos de cooldown
    
    # Simular olho fechado (confiança alta)
    for _ in range(5):
        eye_state.update(0.92)  # Evidência alta
    
    print(f"Confiança: {eye_state.confidence:.3f}")
    print(f"Pode disparar alerta? {not eye_state.is_on_cooldown()}")
    
    # Disparar alerta
    print("\n✅ ALERTA DISPARADO!")
    eye_state.trigger_cooldown()
    print(f"Cooldown ativo? {eye_state.is_on_cooldown()}")
    
    # Tentar disparar de novo imediatamente
    print("\nTentando disparar alerta novamente...")
    print(f"Pode disparar? {not eye_state.is_on_cooldown()} ← NÃO! Bloqueado por 2 segundos")


def test_5_perda_de_face():
    """Teste 5: Face desaparece (is_available=False)"""
    print("\n" + "="*60)
    print("TESTE 5: PERDA TEMPORÁRIA DE FACE")
    print("="*60)
    
    eye_state = MetricState('eye_closure', gain_up=0.15, gain_down=0.05)
    
    # Olho fechado por alguns frames
    print("Frames 1-5: Face detectada, olho fechado")
    for _ in range(5):
        eye_state.update(0.92, is_available=True)
    print(f"  Confiança: {eye_state.confidence:.3f}")
    
    # Face desaparece por 3 frames
    print("\nFrames 6-8: Face DESAPARECEU")
    for frame in range(3):
        eye_state.update(0.0, is_available=False)  # Ignorado, degrada lentamente
        print(f"  Frame {5+frame+1}: confiança={eye_state.confidence:.3f} (degrada lentamente)")
    
    # Face volta
    print("\nFrame 9: Face VOLTOU")
    eye_state.update(0.92, is_available=True)
    print(f"  Confiança: {eye_state.confidence:.3f} (retoma normalmente)")
    
    print(f"\n✅ Avantagem: Não zerou! Mantém contexto.")


def test_6_historico():
    """Teste 6: Análise de histórico temporal"""
    print("\n" + "="*60)
    print("TESTE 6: HISTÓRICO E MÉDIAS TEMPORAIS")
    print("="*60)
    
    eye_state = MetricState('eye_closure', history_size=20)
    
    # Simular 15 frames de olho fechado
    print("Adicionando 15 valores ao histórico...")
    for i in range(15):
        eye_state.update(0.92)
    
    print(f"\n✅ Histórico:")
    print(f"   Tamanho: {len(eye_state.history)} valores")
    print(f"   Primeiros 5: {[f'{v:.3f}' for v in list(eye_state.history)[:5]]}")
    print(f"   Últimos 5: {[f'{v:.3f}' for v in list(eye_state.history)[-5:]]}")
    
    print(f"\n✅ Médias:")
    print(f"   Média de todo histórico: {eye_state.get_history_average():.3f}")
    print(f"   Média dos últimos 3s: {eye_state.get_history_average(seconds=3):.3f}")
    
    print(f"\n✅ Timing:")
    print(f"   Tempo desde último evento: {eye_state.time_since_event():.2f}s")
    

def test_7_debug_info():
    """Teste 7: Informações de debug"""
    print("\n" + "="*60)
    print("TESTE 7: DEBUG INFO")
    print("="*60)
    
    eye_state = MetricState('eye_closure')
    
    for _ in range(8):
        eye_state.update(0.85)
    
    print(f"\n✅ Estado completo:")
    info = eye_state.get_debug_info()
    for key, value in info.items():
        print(f"   {key}: {value}")


def test_8_comparacao_metricas():
    """Teste 8: Comparar comportamento de 2 métricas"""
    print("\n" + "="*60)
    print("TESTE 8: 2 MÉTRICAS SIMULTÂNEAS (EAR + MAR)")
    print("="*60)
    
    eye_state = MetricState('eye_closure', gain_up=0.15, gain_down=0.05)
    mouth_state = MetricState('yawn', gain_up=0.20, gain_down=0.08)
    
    # Simular: olho fechando E boca abrindo
    print("\nFrame | EAR  | Eye Conf | MAR  | Mouth Conf | Status")
    print("-" * 60)
    
    dados = [
        (0.25, 0.30),  # Normal
        (0.20, 0.35),  # Começando
        (0.15, 0.50),  # Progredindo
        (0.10, 0.65),  # Acentuado
        (0.08, 0.75),  # Forte
        (0.07, 0.80),  # Muito forte
    ]
    
    for frame, (ear, mar) in enumerate(dados):
        eye_state.update(1 - ear)
        mouth_state.update((mar - 0.3) / 0.5 if mar > 0.3 else 0)
        
        eye_status = "🟢" if eye_state.confidence < 0.5 else "🟡" if eye_state.confidence < 0.7 else "🔴"
        mouth_status = "🟢" if mouth_state.confidence < 0.5 else "🟡" if mouth_state.confidence < 0.7 else "🔴"
        
        print(f"{frame+1:5d} | {ear:.2f} | {eye_state.confidence:.3f}      | {mar:.2f} | {mouth_state.confidence:.3f}        | {eye_status} + {mouth_status}")
    
    print(f"\n✅ Resultado:")
    print(f"   Olho: {eye_state.confidence:.3f} → {'ALERTA' if eye_state.confidence > 0.7 else 'Normal'}")
    print(f"   Boca: {mouth_state.confidence:.3f} → {'ALERTA' if mouth_state.confidence > 0.7 else 'Normal'}")


def main():
    """Executar todos os testes"""
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  TESTES DO MetricState".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    
    test_1_basico()
    test_2_com_ruido()
    test_3_transicao()
    test_4_cooldown()
    test_5_perda_de_face()
    test_6_historico()
    test_7_debug_info()
    test_8_comparacao_metricas()
    
    print("\n" + "█" * 60)
    print("█" + " TODOS OS TESTES CONCLUÍDOS! ".center(58) + "█")
    print("█" * 60 + "\n")


if __name__ == '__main__':
    main()
