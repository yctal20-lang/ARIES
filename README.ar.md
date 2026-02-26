<div dir="rtl" align="right">

# A.R.I.E.S — نظام الاسترجاع المتقدم والإزالة المدارية

<p align="center">
  <strong>نظام ذكاء اصطناعي مستقل لجمع الحطام الفضائي</strong><br>
  التعلم العميق · التعلم المعزز · دمج المستشعرات · الميكانيكا المدارية
</p>

<p align="center">
  <a href="README.en.md">English</a> · <a href="README.ru.md">Русский</a> · <strong>العربية</strong>
</p>

---

## حول المشروع

A.R.I.E.S (Advanced Retrieval & In-Orbit Elimination System) — نظام مستقل للتحكم في المركبات الفضائية مصمم لإزالة الحطام المداري بشكل فعّال. يجمع النظام بين وكلاء التعلم المعزز، ووحدات الشبكات العصبية، والمحاكاة المبنية على الفيزياء، ولوحة تحكم ويب في الوقت الحقيقي — مكتوب بالكامل بلغة Python.

**عرض مباشر:** منشور على [Render](https://render.com) عبر Flask + Gunicorn.

---

## الميزات الرئيسية

| الميزة | الوصف |
|---|---|
| **بنية أولويات من 4 مستويات** | البقاء → السلامة → حرج للمهمة → تنفيذ المهمة |
| **محاكاة فيزيائية** | ميكانيكا مدارية كبلرية مع اضطراب J2، الضغط الشمسي، السحب الجوي |
| **دمج متعدد المستشعرات** | GPS، IMU، متتبع النجوم مع دمج موزون |
| **وكلاء RL** | SAC (تجنب الاصطدام، المناور)، PPO (إدارة الطاقة) |
| **وحدات عصبية** | CNN كاشف الاصطدام، LSTM مشفر تلقائي (الشذوذ)، TCN (التنبؤ بالحالة)، TFT (التنبؤ بالأعطال)، EfficientNet (التعرف على الحطام)، DETR متتبع |
| **نظام التحمل للأخطاء** | خوارزميات كلاسيكية احتياطية، مؤقتات حراسة، تدهور تلقائي للأوضاع |
| **لوحة تحكم ويب** | واجهة AetherOS داكنة: عرض مداري ثلاثي الأبعاد، قياس عن بُعد، تتبع الحطام، تنبيهات |
| **بيئة Gymnasium** | بيئة مدارية متوافقة مع RL للتدريب |

---

## البنية المعمارية

```
space_debris_ai/
├── core/                              # البنية التحتية الأساسية
│   ├── config.py                      # الإعدادات (Pydantic)
│   ├── base_module.py                 # فئة وحدة IA مجردة (PyTorch)
│   └── message_bus.py                 # ناقل الرسائل بين الوحدات (pub/sub)
│
├── models/                            # وحدات الشبكات العصبية (حسب الأولوية)
│   ├── level1_survival/               # < 100مللي ثانية · 99.999% موثوقية
│   │   ├── collision_avoidance/       # PointNet + CNN + وكيل SAC
│   │   └── navigation/               # EKF + تصحيح عصبي
│   │
│   ├── level2_safety/                 # < 500مللي ثانية · 99.99% موثوقية
│   │   ├── anomaly_detection/         # LSTM مشفر تلقائي + مصنف
│   │   └── energy_management/         # PPO توزيع الطاقة
│   │
│   ├── level3_mission_critical/       # < 1ثانية · 99.9% موثوقية
│   │   ├── state_prediction/          # TCN + دالة خسارة فيزيائية
│   │   ├── early_warning/             # نظام إنذار مبكر قائم على Attention
│   │   ├── sensor_filter/             # مشفر تلقائي لإزالة الضوضاء
│   │   └── failure_prediction/        # Temporal Fusion Transformer (RUL)
│   │
│   └── level4_mission_execution/      # < 2ثانية · 99% موثوقية
│       ├── debris_recognition/        # EfficientNet مصنف متعدد الوسائط
│       ├── manipulator_control/       # SAC التحكم بالذراع الروبوتية
│       ├── object_tracking/           # DETR متتبع متعدد الأجسام
│       ├── precision_maneuvering/     # MPC متحكم المسار
│       └── risk_assessment/           # تقييم مخاطر المهمة
│
├── sensors/                           # واجهات المستشعرات
│   ├── imu.py, lidar.py, camera.py
│   └── fusion.py                      # دمج متعدد المستشعرات
│
├── simulation/                        # بيئة محاكاة مدارية (Gymnasium)
│   ├── physics.py                     # ميكانيكا كبلرية + اضطرابات
│   ├── environment.py                 # OrbitalEnv (Gymnasium API)
│   └── scenarios.py                   # مولد سيناريوهات إجرائي
│
├── safety/                            # آليات التحمل للأخطاء
│   ├── failsafe.py                    # أوضاع احتياطية (Normal → Emergency)
│   └── watchdog.py                    # مراقبة الصحة والمهلات الزمنية
│
├── inference/                         # محرك الاستدلال في الوقت الحقيقي
│   └── mission_controller.py          # المنسق المركزي لجميع الوحدات
│
├── training/                          # سكربتات التدريب والمعايير
├── visualization/                     # لوحات التحكم
│   ├── dashboard.py                   # لوحة تحكم Matplotlib
│   ├── web_server.py                  # لوحة تحكم Flask (إنتاج)
│   ├── templates/index.html           # واجهة AetherOS الداكنة
│   └── static/                        # CSS + JS (Plotly.js 3D)
│
└── tests/                             # مجموعة الاختبارات
```

---

## مستويات الأولوية

| المستوى | الاسم | زمن الاستجابة | الموثوقية | الغرض |
|:---:|---|---|---|---|
| 1 | البقاء | < 100 مللي ثانية | 99.999% | تجنب الاصطدام، الملاحة |
| 2 | السلامة | < 500 مللي ثانية | 99.99% | كشف الشذوذ، إدارة الطاقة |
| 3 | حرج للمهمة | < 1 ثانية | 99.9% | التنبؤ بالحالة، الإنذار المبكر، التنبؤ بالأعطال |
| 4 | تنفيذ المهمة | < 2 ثانية | 99% | التعرف على الحطام، الالتقاط، التتبع |

---

## البدء السريع

### لوحة تحكم الويب (لا تحتاج GPU)

```bash
pip install flask numpy gymnasium gunicorn
python run_web_dashboard.py
```

افتح `http://127.0.0.1:5000` — عرض مداري ثلاثي الأبعاد، قياس عن بُعد، تتبع الحطام، تنبيهات الخطر.

### النظام الكامل (يتطلب PyTorch)

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

## التدريب

```bash
# تجنب الاصطدام (SAC)
python -m space_debris_ai.training.train_collision_avoidance --total-timesteps 1000000

# إدارة الطاقة (PPO)
python -m space_debris_ai.training.train_energy_management --total-timesteps 1000000

# كشف الشذوذ (LSTM مشفر تلقائي)
python -m space_debris_ai.training.train_anomaly_detection --num-epochs 100

# التنبؤ بالحالة (TCN)
python -m space_debris_ai.training.train_state_prediction --num-epochs 100

# التحكم بالمناور (SAC)
python -m space_debris_ai.training.train_manipulator_control --total-timesteps 1000000
```

---

## المحاكاة

بيئة مدارية متوافقة مع Gymnasium:

- ميكانيكا مدارية كبلرية (اضطراب J2، ضغط الإشعاع الشمسي، السحب الجوي)
- حقول حطام قابلة للتكوين (النوع، الحجم، الكتلة، المادة)
- ديناميكا المركبة: الدفع، استهلاك الوقود، التحكم في التوجه
- توليد إجرائي للسيناريوهات مع 8 مستويات صعوبة

---

## لوحة تحكم الويب

واجهة بنمط AetherOS الداكن مع بيانات المهمة في الوقت الحقيقي:

- **تتبع المدار** — أرض ثلاثية الأبعاد + مسار المركبة + حقل الحطام (Plotly.js)
- **القياس عن بُعد** — مكونات الموقع/السرعة عبر الزمن
- **الدمج** — ثقة المستشعرات وسرعة المركبة
- **الموارد** — مستوى الوقود وعدد الحطام
- **تنبيهات الخطر** — تحذيرات الاصطدام، الشذوذ، انخفاض الوقود
- **جدول الحطام** — كل جسم مع الحجم، الكتلة، المادة، المسافة، توصية التخلص

توصيات التخلص تتبع مناهج ESA/NASA: الاستئصال بالليزر، الالتقاط بالشبكة، الحربة/الذراع الروبوتية، الخروج من المدار.

---

## النشر (Render)

| الإعداد | القيمة |
|---|---|
| **وقت التشغيل** | Python 3.11 |
| **أمر البناء** | `pip install -r requirements.txt` |
| **أمر التشغيل** | `gunicorn space_debris_ai.visualization.web_server:app --bind 0.0.0.0:$PORT` |

---

## المقاييس المستهدفة

| المقياس | الهدف |
|---|---|
| تجنب الاصطدام | 100% |
| دقة جمع الحطام | > 95% |
| كفاءة الوقود مقابل الكلاسيكي | +30% |
| التشغيل المستقل | > 90 يوم |
| نجاح الالتقاط (المحاولة الأولى) | > 85% |
| زمن التنبؤ بالأعطال | > 48 ساعة |
| زمن الاستجابة للمستوى 1 | < 100 مللي ثانية |

---

## المكدس التقني

**الأساس:** Python 3.11 · PyTorch 2.x · Gymnasium · Stable-Baselines3

**النماذج:** CNN · LSTM · TCN · Transformer (TFT) · EfficientNet · DETR · PointNet · SAC · PPO · MPC

**المحاكاة:** ميكانيكا كبلرية · اضطراب J2 · دمج مستشعرات EKF

**الويب:** Flask · Gunicorn · Plotly.js · AetherOS CSS

**البنية التحتية:** Render · ONNX Runtime · TensorBoard · W&B

---

## الرخصة

MIT

</div>
