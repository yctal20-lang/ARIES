@echo off
chcp 65001 >nul
echo ============================================================
echo  Разрешение доступа к дашборду с других устройств (порт 5001)
echo  Запустите этот файл ОТ ИМЕНИ АДМИНИСТРАТОРА (ПКМ - Запуск от имени администратора)
echo ============================================================
echo.

netsh advfirewall firewall show rule name="ARIES Dashboard 5001" >nul 2>&1
if %errorlevel% equ 0 (
    echo Правило "ARIES Dashboard 5001" уже есть.
    goto :end
)

netsh advfirewall firewall add rule name="ARIES Dashboard 5001" dir=in action=allow protocol=TCP localport=5001
if %errorlevel% neq 0 (
    echo Ошибка. Запустите файл от имени администратора.
    pause
    exit /b 1
)

echo Правило добавлено. Теперь перезапустите: python run_web_dashboard.py
echo С других устройств откройте в браузере адрес, указанный в консоли (например http://192.168.1.XXX:5001)
:end
echo.
pause
