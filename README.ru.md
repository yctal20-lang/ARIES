# A.R.I.E.S — Продвинутая Система Извлечения и Орбитальной Утилизации

<p align="center">
  <strong>Автономная ИИ-система для сбора космического мусора</strong><br>
  Глубокое обучение · Обучение с подкреплением · Сенсорная фузия · Орбитальная механика
</p>

<p align="center">
  <a href="README.en.md">English</a> · <strong>Русский</strong>
</p>

---

## О проекте

A.R.I.E.S (Advanced Retrieval & In-Orbit Elimination System) — автономная система управления космическим аппаратом для активного удаления орбитального мусора. Система объединяет агентов обучения с подкреплением, нейросетевые модули, физически информированную симуляцию и веб-дашборд реального времени — полностью написана на Python.

**Демо:** развёрнуто на [Render](https://render.com) через Flask + Gunicorn.

---

## Ключевые возможности

| Возможность | Описание |
|---|---|
| **4-уровневая приоритетная архитектура** | Выживание → Безопасность → Миссия (критич.) → Выполнение миссии |
| **Физическая симуляция** | Кеплеровская орбитальная механика с J2-пертурбацией, солнечным давлением, атмосферным торможением |
| **Мультисенсорная фузия** | GPS, IMU, Star Tracker со взвешенным объединением |
| **RL-агенты** | SAC (уклонение от столкновений, манипулятор), PPO (управление энергией) |
| **Нейросетевые модули** | CNN детектор столкновений, LSTM автоэнкодер (аномалии), TCN (предсказание состояния), TFT (прогноз отказов), EfficientNet (распознавание мусора), DETR трекер |
| **Система отказоустойчивости** | Классические алгоритмы как фолбэки, watchdog-таймеры, автоматическая деградация режимов |
| **Веб-дашборд** | Тёмная тема AetherOS: 3D орбита, телеметрия, отслеживание мусора, предупреждения |
| **Gymnasium-среда** | Полноценная RL-совместимая орбитальная среда для обучения |

---

## Архитектура

```
space_debris_ai/
├── core/                              # Базовая инфраструктура
│   ├── config.py                      # Конфигурация (Pydantic)
│   ├── base_module.py                 # Абстрактный класс ИИ-модуля (PyTorch)
│   └── message_bus.py                 # Межмодульная шина сообщений (pub/sub)
│
├── models/                            # Нейросетевые модули (по приоритету)
│   ├── level1_survival/               # < 100мс · 99.999% надёжность
│   │   ├── collision_avoidance/       # PointNet + CNN + SAC-агент
│   │   └── navigation/               # EKF + нейрокоррекция
│   │
│   ├── level2_safety/                 # < 500мс · 99.99% надёжность
│   │   ├── anomaly_detection/         # LSTM-автоэнкодер + классификатор
│   │   └── energy_management/         # PPO-управление питанием
│   │
│   ├── level3_mission_critical/       # < 1с · 99.9% надёжность
│   │   ├── state_prediction/          # TCN + физически обоснованная функция потерь
│   │   ├── early_warning/             # Attention-система раннего предупреждения
│   │   ├── sensor_filter/             # Шумоподавляющий автоэнкодер
│   │   └── failure_prediction/        # Temporal Fusion Transformer (RUL)
│   │
│   └── level4_mission_execution/      # < 2с · 99% надёжность
│       ├── debris_recognition/        # EfficientNet мультимодальный классификатор
│       ├── manipulator_control/       # SAC-управление роботизированной рукой
│       ├── object_tracking/           # DETR-подобный мульти-объектный трекер
│       ├── precision_maneuvering/     # MPC-контроллер траектории
│       └── risk_assessment/           # Оценка рисков миссии
│
├── sensors/                           # Сенсорные интерфейсы
│   ├── imu.py, lidar.py, camera.py
│   └── fusion.py                      # Мультисенсорное объединение
│
├── simulation/                        # Gymnasium-среда орбитальной симуляции
│   ├── physics.py                     # Кеплеровская механика + пертурбации
│   ├── environment.py                 # OrbitalEnv (Gymnasium API)
│   └── scenarios.py                   # Процедурный генератор сценариев
│
├── safety/                            # Отказоустойчивость
│   ├── failsafe.py                    # Режимы фолбэка (Normal → Emergency)
│   └── watchdog.py                    # Мониторинг здоровья и таймауты
│
├── inference/                         # Движок инференса реального времени
│   └── mission_controller.py          # Центральный оркестратор всех модулей
│
├── training/                          # Скрипты обучения и бенчмарки
├── visualization/                     # Дашборды
│   ├── dashboard.py                   # Matplotlib-дашборд миссии
│   ├── web_server.py                  # Flask веб-дашборд (продакшен)
│   ├── templates/index.html           # Тёмная тема AetherOS
│   └── static/                        # CSS + JS (Plotly.js 3D-графики)
│
└── tests/                             # Тесты
```

---

## Уровни приоритета

| Уровень | Название | Задержка | Надёжность | Назначение |
|:---:|---|---|---|---|
| 1 | Выживание | < 100 мс | 99.999% | Уклонение от столкновений, навигация |
| 2 | Безопасность | < 500 мс | 99.99% | Обнаружение аномалий, управление питанием |
| 3 | Критичные для миссии | < 1 с | 99.9% | Предсказание состояния, раннее предупреждение, прогноз отказов |
| 4 | Выполнение миссии | < 2 с | 99% | Распознавание мусора, захват, отслеживание |

---

## Быстрый старт

### Веб-дашборд (GPU не нужен)

```bash
pip install flask numpy gymnasium gunicorn
python run_web_dashboard.py
```

Откройте `http://127.0.0.1:5000` — 3D-орбита, телеметрия, отслеживание мусора, предупреждения об опасности.

### Полная система (нужен PyTorch)

```bash
pip install -r space_debris_ai/requirements.txt
```

```python
from space_debris_ai import SystemConfig
from space_debris_ai.simulation import OrbitalEnv
from space_debris_ai.inference import MissionController

config = SystemConfig(mission_name="debris_collection_001")
env = OrbitalEnv()
obs, info = env.reset()

controller = MissionController(config)
controller.start()

for step in range(1000):
    result = controller.step(
        {"position": obs[:3], "velocity": obs[3:6], "attitude": obs[6:10]},
        dt=0.1,
    )
    obs, reward, done, truncated, info = env.step(result["commands"])
    if done or truncated:
        break

controller.stop()
env.close()
```

---

## Обучение

```bash
# Уклонение от столкновений (SAC)
python -m space_debris_ai.training.train_collision_avoidance --total-timesteps 1000000

# Управление энергией (PPO)
python -m space_debris_ai.training.train_energy_management --total-timesteps 1000000

# Обнаружение аномалий (LSTM-автоэнкодер)
python -m space_debris_ai.training.train_anomaly_detection --num-epochs 100

# Предсказание состояния (TCN)
python -m space_debris_ai.training.train_state_prediction --num-epochs 100

# Управление манипулятором (SAC)
python -m space_debris_ai.training.train_manipulator_control --total-timesteps 1000000
```

---

## Симуляция

Gymnasium-совместимая орбитальная среда:

- Кеплеровская орбитальная механика (J2-пертурбация, солнечное давление, атмосферное торможение)
- Настраиваемые поля мусора (тип, размер, масса, материал)
- Динамика аппарата: тяга, расход топлива, управление ориентацией
- Процедурная генерация сценариев с 8 уровнями сложности

---

## Веб-дашборд

Интерфейс в стиле AetherOS с данными миссии в реальном времени:

- **Orbit Track** — 3D Земля + траектория аппарата + поле мусора (Plotly.js)
- **Телеметрия** — компоненты позиции/скорости во времени
- **Фузия** — уверенность сенсоров и скорость аппарата
- **Ресурсы** — уровень топлива и количество обломков
- **Предупреждения** — столкновения, аномалии, низкое топливо
- **Таблица мусора** — каждый объект с размером, массой, материалом, дистанцией, рекомендацией по утилизации

Рекомендации по утилизации по стандартам ESA/NASA: лазерная абляция, сетевой захват, гарпун/робот-рука, сведение с орбиты.

---

## Деплой (Render)

| Параметр | Значение |
|---|---|
| **Runtime** | Python 3.11 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn space_debris_ai.visualization.web_server:app --bind 0.0.0.0:$PORT` |

---

## Целевые метрики

| Метрика | Цель |
|---|---|
| Уклонение от столкновений | 100% |
| Точность сбора мусора | > 95% |
| Эффективность топлива vs классика | +30% |
| Автономная работа | > 90 дней |
| Успех захвата (первая попытка) | > 85% |
| Прогноз отказов (упреждение) | > 48 часов |
| Задержка модулей уровня 1 | < 100 мс |

---

## Стек технологий

**Ядро:** Python 3.11 · PyTorch 2.x · Gymnasium · Stable-Baselines3

**Модели:** CNN · LSTM · TCN · Transformer (TFT) · EfficientNet · DETR · PointNet · SAC · PPO · MPC

**Симуляция:** Кеплеровская механика · J2-пертурбация · EKF-фузия

**Веб:** Flask · Gunicorn · Plotly.js · AetherOS CSS

**Инфра:** Render · ONNX Runtime · TensorBoard · W&B

---

## Лицензия

MIT
