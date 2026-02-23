# A.R.I.E.S Web Dashboard

Запуск веб-дашборда в стиле AetherOS.

## Установка зависимостей

```bash
pip install flask>=3.0.0
```

(или установите все зависимости из requirements.txt)

## Запуск

```bash
cd C:\Users\Asus\Desktop\CURSOR
python run_web_dashboard.py
```

Откроется сервер на `http://127.0.0.1:5000`

Откройте браузер и перейдите по этому адресу.

## Возможности

- 🌍 3D визуализация орбиты (Земля + траектория + обломки)
- 📊 Телеметрия (X, Y, Z компоненты позиции)
- 🚀 Скорость и уверенность fusion
- ⛽ Ресурсы (топливо и количество обломков)
- 🎨 Интерактивные графики Plotly.js
- 🌑 Тёмная тема AetherOS
- 📱 Адаптивный дизайн

## Структура

- `visualization/web_server.py` — Flask сервер
- `visualization/templates/index.html` — HTML шаблон
- `visualization/static/js/dashboard.js` — JavaScript визуализация
- `run_web_dashboard.py` — Скрипт запуска
