# CFNN — Plan de Ejecución Completo

## Objetivo Final
Publicar en HuggingFace un modelo + Space interactivo de una arquitectura neural novel inspirada en ingeniería química, con paper/blog post explicativo y benchmarks.

---

## FASE 0: Setup (Hoy — Viernes 13 Marzo)
**Duración:** 1-2 horas  
**Objetivo:** Tener el repo listo y el entorno configurado.

### Tareas:
- [ ] Crear repo en GitHub: `DanielRegaladoUMiami/counterflow-nn`
- [ ] Estructura inicial de carpetas (src/, experiments/, notebooks/, docs/)
- [ ] Mover el documento técnico `CFNN_Technical_Documentation.md` a `docs/`
- [ ] Setup `requirements.txt`: torch, numpy, scikit-learn, matplotlib, gradio
- [ ] Crear `README.md` con descripción corta + diagrama ASCII de la arquitectura
- [ ] Crear repo en HuggingFace: `DanielRegaladoCardoso/counterflow-nn`

### Entregable:
Repo en GitHub con estructura vacía y documentación técnica.

---

## FASE 1: Core Architecture (Sábado 14 — Domingo 15)
**Duración:** 2 días  
**Objetivo:** Implementar CFNN-A (absorción) y verificar que entrena.

### Día 1 (Sábado): Las piezas

#### Mañana: `src/plates.py`
- [ ] Implementar `CounterFlowPlate` (un plato):
  - Equilibrium function E(l) → gas-equivalent
  - Driving force: δ = g - E(l)
  - Transfer function: Δ = α · T(δ)
  - Update: g_new = g - Δ, l_new = l + Δ
- [ ] Test unitario: verificar conservación (Δg + Δl = 0)
- [ ] Test unitario: verificar dimensiones de entrada/salida

#### Tarde: `src/network.py`
- [ ] Implementar `CounterFlowNetwork` (torre completa):
  - Gas encoder: input → g₀
  - Liquid initialization: zeros o context → l_{N+1}
  - Approach C: sweeps alternados (líquido baja, gas sube)
  - Output head: concatenar g_N y l_1 → predicción
- [ ] Parámetro `n_sweeps` configurable (default=2)
- [ ] Parámetro `share_plates` (True = pesos compartidos, False = únicos)
- [ ] Forward pass completo funcionando sin errores

#### Noche: Primer entrenamiento
- [ ] Dataset sintético: clasificación de espirales 2D (sklearn.make_moons)
- [ ] Entrenar CFNN-A con 5 platos, d_gas=32, d_liquid=32
- [ ] Verificar que el loss baja → la red aprende
- [ ] Si no aprende: debug (gradientes, driving force, etc.)

### Día 2 (Domingo): Baselines + comparación inicial

#### Mañana: `src/activations.py`
- [ ] Implementar activaciones ChemE:
  - `MichaelisMenten(x)` = V_max · x / (K_m + |x|)
  - `Arrhenius(x)` = x · exp(-E_a / (|x| + ε))
  - `HillActivation(x)` = x^n / (K^n + x^n)
- [ ] Probar cada una como reemplazo de la activación en los platos

#### Tarde: Baselines
- [ ] Implementar MLP baseline (mismo número de parámetros que CFNN)
- [ ] Implementar ResNet-MLP baseline (MLP con skip connections)
- [ ] Script `experiments/compare_baselines.py`:
  - Mismo dataset (moons, circles, XOR)
  - Mismo budget de parámetros
  - 5 seeds por experimento
  - Reportar accuracy media ± std

#### Noche: Primer análisis
- [ ] ¿CFNN aprende? ¿Es competitivo con MLP?
- [ ] Si no: ajustar hiperparámetros (n_sweeps, α inicial, learning rate)
- [ ] Si sí: pasar a Fase 2

### Entregable Fase 1:
- CFNN-A entrenando y aprendiendo en datasets sintéticos
- Comparación numérica básica contra MLP y ResNet
- Código limpio en src/

---

## FASE 2: Destilación + Benchmarks Serios (Lunes 16 — Martes 17)
**Duración:** 2 días  
**Objetivo:** Implementar CFNN-D y probar en datasets reales.

### Día 3 (Lunes): CFNN-D

#### Mañana: Upgrade a destilación
- [ ] Implementar `DistillationPlate` en `src/plates.py`:
  - Transfer bidireccional (sin tanh, delta puede ser + o -)
  - Feed plate con inyección y φ aprendible
- [ ] Implementar `DistillationNetwork` en `src/network.py`:
  - Secciones de rectificación y stripping
  - Feed plate en posición configurable
  - Reflux ratio como parámetro
  - Dos outputs: distillate (g_N) y bottoms (l_1)

#### Tarde: Benchmarks Tier 2 (Tabular)
- [ ] `experiments/tier2_tabular.py`:
  - Datasets: Iris, Wine, Breast Cancer, California Housing
  - Modelos: CFNN-A, CFNN-D, MLP, ResNet-MLP
  - Métricas: Accuracy (clasificación) o RMSE (regresión)
  - 5 seeds, reportar media ± std
  - Guardar resultados en CSV

#### Noche: Análisis
- [ ] Tabla comparativa de resultados
- [ ] ¿CFNN-D mejora sobre CFNN-A?
- [ ] ¿En qué datasets gana/pierde vs baselines?

### Día 4 (Martes): Benchmarks Tier 3 + Diagnósticos

#### Mañana: Tier 3 (MNIST)
- [ ] `experiments/tier3_images.py`:
  - MNIST y FashionMNIST (flattened 784-dim)
  - Mismos modelos
  - Entrenar por 50 epochs, reportar test accuracy

#### Tarde: `src/diagnostics.py`
- [ ] Implementar tracking de:
  - Driving force por plato: ||δ_n|| a lo largo del entrenamiento
  - Transfer amount por plato: ||Δ_n||
  - Dirección de transfer (para CFNN-D): % platos rectificando vs stripping
  - Gradient norms por plato
  - Damköhler number estimado
- [ ] Guardar logs para visualización

#### Noche: Decisión Go/No-Go
- [ ] Revisar todos los resultados hasta ahora
- [ ] Decidir: ¿CFNN tiene mérito? ¿En qué tareas?
- [ ] Si sí → Fase 3 (visualización + HF)
- [ ] Si no → Pivotar a las fortalezas encontradas (interpretabilidad, denoising, etc.)

### Entregable Fase 2:
- CFNN-D implementado y funcional
- Tabla de benchmarks en 6+ datasets vs 2+ baselines
- Diagnósticos de la dinámica interna del network

---

## FASE 3: Visualización + Análisis (Miércoles 18)
**Duración:** 1 día  
**Objetivo:** Las visualizaciones que hacen al proyecto único.

### Mañana: `src/visualization.py`

- [ ] **McCabe-Thiele Neural Plot:**
  - Eje X: ||l_n|| (estado del líquido en cada plato)
  - Eje Y: ||g_n|| (estado del gas en cada plato)
  - Curva: equilibrium function E aprendida
  - Escalones: trayectoria real del gas/líquido por los platos
  - Debe verse como un McCabe-Thiele clásico pero generado por la red

- [ ] **Concentration Profile:**
  - Eje X: número de plato (0 a N)
  - Eje Y: norma de features
  - Dos líneas: ||g_n|| descendiendo, ||l_n|| ascendiendo
  - Analogía visual directa con perfil de concentración en torre

- [ ] **Transfer Direction Heatmap** (CFNN-D):
  - Cada plato coloreado: rojo = rectificando, azul = stripping
  - Mostrar dónde la red decide extraer vs inyectar

- [ ] **Driving Force Profile:**
  - ||δ_n|| vs plato
  - Identificar si hay "pinch points" (driving force → 0)

### Tarde: `notebooks/01_concept_and_theory.ipynb`

- [ ] Notebook explicativo completo:
  1. Diagrama de torre de absorción (matplotlib)
  2. Ecuaciones de ChemE y su mapeo neural
  3. Arquitectura CFNN paso a paso con ejemplo numérico
  4. Entrenamiento en dataset simple con visualizaciones en vivo
  5. McCabe-Thiele neural generado
  6. Comparación visual CFNN vs MLP (decision boundaries)

### Entregable Fase 3:
- 4+ visualizaciones únicas inspiradas en ChemE
- Notebook tutorial completo y autocontenido

---

## FASE 4: HuggingFace Deployment (Jueves 19)
**Duración:** 1 día  
**Objetivo:** Todo publicado y accesible.

### Mañana: Gradio Space (`app.py`)

- [ ] Tab 1 — "Learn": Explicación interactiva de la arquitectura
  - Slider: número de platos
  - Slider: dimensión gas / líquido
  - Visualización en vivo de la arquitectura
  - Analogía ChemE explicada

- [ ] Tab 2 — "Train & Compare": Demo funcional
  - Dropdown: seleccionar dataset (moons, circles, iris, etc.)
  - Botón: entrenar CFNN vs MLP
  - Gráficas: loss curves, accuracy, decision boundaries
  - Tabla: resultados comparativos

- [ ] Tab 3 — "Analyze": Las visualizaciones ChemE
  - McCabe-Thiele neural (post-entrenamiento)
  - Concentration profiles
  - Driving force analysis
  - Transfer direction map

### Tarde: Publicación

- [ ] Push código a GitHub con README completo
- [ ] Push modelo entrenado a HuggingFace Hub
- [ ] Deploy Space en HuggingFace
- [ ] Escribir HuggingFace blog post / model card:
  - Contexto: ingeniero químico + data scientist
  - La analogía explicada de forma accesible
  - Resultados clave
  - Link al paper técnico y código

### Noche: QA
- [ ] Probar el Space end-to-end
- [ ] Verificar que todos los links funcionan
- [ ] Pedir feedback a alguien (compañero, profesor)

### Entregable Fase 4:
- HF Space live y funcional
- Modelo publicado en HF Hub
- Repo de GitHub completo con docs, code, experiments
- Blog post / model card

---

## FASE 5: Polish + Stretch Goals (Viernes 20)
**Duración:** 1 día (si hay tiempo)  
**Objetivo:** Refinamiento y extras.

### Opcionales (priorizar según resultados):
- [ ] Tier 4 experiments: denoising autoencoder con CFNN
- [ ] Multi-task learning con CFNN-D (distillate = tarea A, bottoms = tarea B)
- [ ] Continuous-depth version (Neural ODE counterflow)
- [ ] Publicar las activaciones ChemE como paquete separado en PyPI
- [ ] Video explicativo corto (2-3 min) para LinkedIn/Twitter
- [ ] Conectar con profesor de ChemE o ML para feedback académico

---

## Checklist de Entregables Finales

### Código
- [ ] `src/plates.py` — CounterFlowPlate + DistillationPlate
- [ ] `src/network.py` — CFNN-A + CFNN-D
- [ ] `src/activations.py` — Michaelis-Menten, Arrhenius, Hill
- [ ] `src/diagnostics.py` — Damköhler, driving force, gradient tracking
- [ ] `src/visualization.py` — McCabe-Thiele neural + profiles
- [ ] `experiments/` — Tier 1-3 benchmarks con resultados reproducibles
- [ ] `app.py` — Gradio Space funcional

### Documentación
- [ ] `docs/CFNN_Technical_Documentation.md` — Paper técnico completo
- [ ] `README.md` — Overview accesible con diagramas
- [ ] `notebooks/01_concept_and_theory.ipynb` — Tutorial visual
- [ ] HuggingFace Model Card
- [ ] HuggingFace Blog Post (opcional pero ideal)

### Resultados
- [ ] Tabla de benchmarks: CFNN-A, CFNN-D vs MLP, ResNet en 6+ datasets
- [ ] Visualización McCabe-Thiele neural (LA imagen icónica del proyecto)
- [ ] Análisis de driving force y gradient flow
- [ ] Assessment honesto: dónde CFNN gana, dónde pierde, y por qué

### Publicación
- [ ] GitHub repo público
- [ ] HuggingFace Space live
- [ ] Modelo en HuggingFace Hub

---

## Criterios de Éxito

### Mínimo viable (must have):
1. CFNN-A entrena y aprende en al menos 3 datasets
2. Comparación cuantitativa contra MLP baseline
3. Al menos 1 visualización McCabe-Thiele neural
4. Código publicado en GitHub

### Bueno (should have):
5. CFNN-D implementado y comparado
6. Benchmarks en 6+ datasets con 5 seeds
7. HF Space funcional
8. Documento técnico completo

### Excelente (nice to have):
9. CFNN supera baselines en al menos 1 tarea
10. Activaciones ChemE muestran beneficio
11. Blog post publicado
12. Feedback de al menos 1 persona externa

---

## Riesgos y Pivots

| Si pasa esto... | Entonces... |
|---|---|
| CFNN no aprende en nada | Simplificar: quitar conservación, usar 1 sweep, aumentar α |
| CFNN aprende pero nunca gana | Enfocar en interpretabilidad — las visualizaciones son el valor |
| CFNN-D es inestable | Quedarse con CFNN-A que es más simple y estable |
| No da tiempo para HF Space | Publicar solo el repo + notebook como MVP |
| Los resultados son ambiguos | Ser honesto en el writeup — "work in progress" es válido |

---

## Notas Personales

- Este proyecto combina mi BS en Chemical Engineering con mi MSBA
- Las ecuaciones vienen de Treybal (Mass Transfer) y Fogler (Reactor Design)
- Berkeley Madonna era el software donde modelaba CSTRs en serie y paralelo
- La presentación de absorción de gases es de mi curso de Transferencia de Masa
- El contexto de CBC (distribución de bebidas en Centroamérica) inspira las analogías de supply chain, pero el proyecto es de ML puro
- Objetivo: que esto sea parte del portfolio para el job search post-graduación (Mayo 2026)
