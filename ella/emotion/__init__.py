"""Ella Emotion Engine — persistent, personality-mediated emotional state.

Grounded in:
  - PAD Model (Mehrabian & Russell, 1974): Pleasure-Arousal-Dominance
  - Bosse / Pereira-Paiva Generic Contagion Model (ACII 2011)
  - Doherty's Emotional Contagion Scale (ECS, 1997)

Public surface:
  models  — data structures (EmotionProfile, AgentState, UserState, PersonalityTraits)
  store   — async MySQL persistence (EmotionStore)
  engine  — contagion, self-update, decay algorithms
"""
