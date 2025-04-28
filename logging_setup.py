# logging_setup.py
import logging
from datetime import datetime, timedelta
import os
import time
from typing import (
    Optional,
    Union,
    TextIO,	
    Callable,
    Any,
    TypeVar,
    Dict,
    List,
    cast
)
from functools import wraps

# Тип для generic-функций
F = TypeVar('F', bound=Callable[..., Any])

class TimedLogger:
    """Логгер с замером времени выполнения операций.
    
    Attributes:
        _logger (logging.Logger): Базовый логгер.
        _start_times (Dict[str, float]): Словарь для хранения времени начала операций.
    """
    
    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        level: Union[int, str] = logging.INFO,
        console_output: bool = True,
        file_output: bool = True
    ) -> None:
        """Инициализация TimedLogger.
        
        Args:
            name: Имя логгера (обычно __name__ или имя класса).
            log_dir: Директория для хранения логов. По умолчанию "logs".
            level: Уровень логирования. По умолчанию logging.INFO.
            console_output: Флаг вывода в консоль. По умолчанию True.
            file_output: Флаг записи в файл. По умолчанию True.
        """
        self._logger: logging.Logger = self._configure_logger(
            name, log_dir, level, console_output, file_output
        )
        self._start_times: Dict[str, float] = {}
    
    def get_log_path(self) -> str:
        """Пример метода, использующего self.log_dir"""
        return os.path.abspath(self.log_dir)
    
    def _configure_logger(
        self,
        name: str,
        log_dir: str,
        level: Union[int, str],
        console_output: bool,
        file_output: bool
    ) -> logging.Logger:
        """Конфигурация базового логгера.
        
        Args:
            name: Имя логгера.
            log_dir: Директория для логов.
            level: Уровень логирования.
            console_output: Включить вывод в консоль.
            file_output: Включить запись в файл.
            
        Returns:
            Настроенный экземпляр logging.Logger.
        """
        os.makedirs(log_dir, exist_ok=True)
        current_date: str = datetime.now().strftime("%Y-%m-%d")
        log_file: str = f"{log_dir}/{name}_{current_date}.log"

        logger: logging.Logger = logging.getLogger(name)
        logger.setLevel(level)
        
        if not logger.handlers:
            formatter: logging.Formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s | [%(duration)s]"
            )
            
            handlers: List[logging.Handler] = []
            
            if file_output:
                file_handler: logging.FileHandler = logging.FileHandler(
                    log_file, 
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                handlers.append(file_handler)
            
            if console_output:
                console_handler: logging.StreamHandler[TextIO] = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                handlers.append(console_handler)
            
            for handler in handlers:
                logger.addHandler(handler)
        
        return logger
    
    def log_operation(self, message: str, level: str = "info") -> Callable[[F], F]:
        """Декоратор для логирования времени выполнения функции.
        
        Args:
            message: Описание операции.
            level: Уровень логирования ('info', 'warning', 'error').
            
        Returns:
            Декоратор для функции.
        """
        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time: float = time.time()
                self._logger.info(
                    f"🚀 Начало: {message}", 
                    extra={"duration": "N/A"}
                )
                
                try:
                    result: Any = func(*args, **kwargs)
                    duration: timedelta = timedelta(seconds=time.time() - start_time)
                    
                    log_method = getattr(self._logger, level)
                    log_method(
                        f"✅ Успешно: {message}", 
                        extra={"duration": str(duration)}
                    )
                    return result
                except Exception as e:
                    duration = timedelta(seconds=time.time() - start_time)
                    self._logger.error(
                        f"❌ Ошибка: {message} ({str(e)})", 
                        extra={"duration": str(duration)},
                        exc_info=True
                    )
                    raise
            
            return cast(F, wrapper)
        return decorator
    
    def start_timer(self, operation_name: str) -> None:
        """Начать замер времени для операции.
        
        Args:
            operation_name: Уникальное имя операции.
            
        Raises:
            ValueError: Если операция с таким именем уже начата.
        """
        if operation_name in self._start_times:
            raise ValueError(f"Операция '{operation_name}' уже начата")
        self._start_times[operation_name] = time.time()
        self._logger.info(
            f"⏳ Начало операции: {operation_name}",
            extra={"duration": "N/A"}
        )
    
    def end_timer(self, operation_name: str) -> timedelta:
        """Завершить замер времени и записать результат.
        
        Args:
            operation_name: Имя начатой операции.
            
        Returns:
            Продолжительность выполнения операции.
            
        Raises:
            KeyError: Если операция не была начата.
        """
        if operation_name not in self._start_times:
            raise KeyError(f"Операция '{operation_name}' не была начата")
        
        duration: timedelta = timedelta(seconds=time.time() - self._start_times[operation_name])
        self._logger.info(
            f"⌛ Завершено: {operation_name}",
            extra={"duration": str(duration)}
        )
        del self._start_times[operation_name]
        return duration
    
    def info(self, message: str) -> None:
        """Логирование информационного сообщения.
        
        Args:
            message: Текст сообщения.
        """
        self._logger.info(message, extra={"duration": "N/A"})
    
    def warning(self, message: str) -> None:
        """Логирование предупреждения.
        
        Args:
            message: Текст сообщения.
        """
        self._logger.warning(message, extra={"duration": "N/A"})
    
    def error(self, message: str, exc_info: bool = True) -> None:
        """Логирование ошибки.
        
        Args:
            message: Текст сообщения.
            exc_info: Включать ли информацию об исключении. По умолчанию True.
        """
        self._logger.error(
            message, 
            extra={"duration": "N/A"}, 
            exc_info=exc_info
        )