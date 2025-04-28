import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Pattern

import pdfplumber
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from logging_setup import TimedLogger
from langdetect import detect, DetectorFactory

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from functools import lru_cache

logging.getLogger('pdfminer').setLevel(logging.ERROR)

class pdf_converter:
    """PDF-ридер для LlamaIndex с расширенными возможностями"""
    
    # Регулярные выражения для очистки текста (скомпилированы заранее для производительности)
    FIGURE_PATTERN = re.compile(r'(?im)^.*Рис\..*$')
    TABLE_PATTERN = re.compile(r'(?i)^\s*Табл\..*?(\n|\Z)', flags=re.MULTILINE)
    TOC_PATTERN = re.compile(r'(?i)(Содержание|Оглавление|Contents).*?(\n\n|\Z)', flags=re.DOTALL)
    LIST_MARKERS_PATTERN = re.compile(r"[•▪➢✧]")
    
    # Паттерны для определения таблиц
    NON_TABLE_PATTERNS = [
        re.compile(r'Внимание!'),
        re.compile(r"(?:\n|\r\n)\s*Примечание."),
        re.compile(r"«"),
        re.compile(r'^\s*[-•*]'),  # Строки начинаются с маркеров списка
        re.compile(r':\s*\n'),     # Двоеточие с переносом
        re.compile(r';\s*\n')      # Точка с запятой с переносом
    ]
    
    # Паттерны для определения оглавления
    TOC_PATTERNS = [
        re.compile(r'оглавление'),
        re.compile(r'содержание'),
        re.compile(r'contents'),
        re.compile(r'table of contents'),
        re.compile(r'^\s*\d+\s*\.\s*\w+'),       # Строки, начинающиеся с цифры и точки
        re.compile(r'^\s*глава\s+\d+'),          # Строки "Глава 1"
        re.compile(r'^\s*\d+\s+\.\.\.\s+\d+'),   # Строки типа "1 ...... 15"
        re.compile(r'^\s*\d+\s+[a-яё]\s*\.'),    # Строки типа "1 а. Введение"
        re.compile(r'^\s*[ivx]+\s*\.'),          # Римские цифры с точкой
        re.compile(r'^\s*приложение\s+[a-я\d]+') # Приложения
    ]
    
     # Паттерны для определения типа документа
    DOC_TYPE_PATTERNS: List[Pattern] = [
        re.compile(r"руководство\s+администратора", re.IGNORECASE),
        re.compile(r"техническое\s+руководство", re.IGNORECASE),
        re.compile(r"инструкция\s+пользователя", re.IGNORECASE),
        re.compile(r"руководство\s+оператора", re.IGNORECASE)
    ]
    
    # Паттерн для поиска слова "руководство" в строке
    GUIDE_PATTERN = re.compile(r"руководство", re.IGNORECASE)

    # Паттерн для поиска аннотации
    ANNOTATION_PATTERN = re.compile(r"аннотация", re.IGNORECASE)
    
    def __init__(
        self,
        include_tables: bool = True,  
        include_metadata: bool = True,  
        default_top_margin: int = 50,
        default_bottom_margin: int = 60,
        table_settings: Optional[Dict[str, Any]] = None,
        default_log_dir = "logs"
    ):
        """
        Инициализирует конвертер PDF.
        
        Args:
            include_tables: Включать ли таблицы в результат (по умолчанию True)
            include_metadata: Включать ли метаданные (по умолчанию True)
            default_top_margin: Верхний отступ страницы в пунктах (по умолчанию 50)
            default_bottom_margin: Нижний отступ страницы в пунктах (по умолчанию 60)
            table_settings: Настройки для извлечения таблиц (по умолчанию None)
            default_log_dir: Директория для хранения log-файлов
        """
        self.logger = TimedLogger(self.__class__.__name__, log_dir=default_log_dir)
        self.doc_tables = [] # Таблицы документа
        self.include_tables = include_tables
        self.include_metadata = include_metadata
        self.default_top_margin = default_top_margin
        self.default_bottom_margin = default_bottom_margin
        self.settings = table_settings or { 
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_y_tolerance": 5,
            "text_x_tolerance": 3,  # Меньший допуск для текста
            "text_y_tolerance": 3,
            "snap_tolerance": 3,
            "join_tolerance": 3  # Минимальное расстояние для объединения элементов
        }
        self.root_doc_id  = None # Корневой идентификатор документа
  
    def clean_pdf_text(self, text: str) -> str:
        """Очищает текст, извлеченный из PDF, от специфичных для технической документации артефактов.
    
        Выполняет следующие преобразования:
        1. Удаляет строки, содержащие упоминания рисунков (например, "Рис. 1. Схема системы")
        2. Удаляет блоки, относящиеся к таблицам (начинающиеся с "Табл.")
        3. Удаляет разделы оглавления/содержания
        4. Нормализует специальные символы маркеров списков к стандартному дефису

        Args:
            text: Входной текст для очистки. Если передана пустая строка или None,
                  метод немедленно возвращает пустую строку.

        Returns:
            Очищенный текст с удаленными техническими артефактами. Гарантированно
            возвращает строку (может быть пустой).

        Raises:
            TypeError: Если аргумент text не является строкой.

        Examples:
            >>> converter = UserManualPDFConverter()
            >>> converter.clean_pdf_text("Рис. 1. Схема\\nОсновной текст")
            'Основной текст'
        
            >>> converter.clean_pdf_text("Табл. 1. Параметры\\n• Пункт 1")
            '- Пункт 1'
        
            >>> converter.clean_pdf_text(None)
            ''
        """
        if not isinstance(text, str):
             raise TypeError(f"Ожидается строковый аргумент, получен {type(text).__name__}")
        
        if not text:
            return ""

        # Последовательное применение предкомпилированных регулярных выражений
        text = self.FIGURE_PATTERN.sub('', text).strip()  # Удаление ссылок на рисунки
        text = self.TABLE_PATTERN.sub('', text)          # Удаление ссылок на таблицы
        text = self.TOC_PATTERN.sub('', text)            # Удаление оглавлений
        text = self.LIST_MARKERS_PATTERN.sub("-", text)  # Нормализация маркеров списков
        
        return text
        
    def is_real_table(self, table_data: List[List[str]]) -> bool:
        """Определяет, является ли переданная структура данных валидной таблицей документа.
    
        Проверяет таблицу по следующим критериям:
        1. Минимальный размер таблицы (2 строки и 2 столбца)
        2. Отсутствие в тексте таблицы шаблонов, характерных для не-табличных данных
           (примечания, маркеры списков, предупреждения и т.д.)

        Args:
            table_data: Двумерный список строк, представляющий потенциальную таблицу.
                        Каждый внутренний список соответствует строке таблицы.

        Returns:
            True если переданные данные соответствуют критериям таблицы,
            False в противном случае.

        Raises:
            TypeError: Если table_data не является двумерным списком строк.
            ValueError: Если table_data пуст или содержит не-строковые элементы.

        Examples:
            >>> converter = UserManualPDFConverter()
            >>> valid_table = [["Header1", "Header2"], ["Data1", "Data2"]]
            >>> converter.is_real_table(valid_table)
            True
        
            >>> invalid_table = [["Примечание:"], ["Данные не представлены"]]
            >>> converter.is_real_table(invalid_table)
            False
        
            >>> converter.is_real_table([])
            False
        """
        # Проверка типов и базовых условий
        if not isinstance(table_data, list):
            raise TypeError(f"Ожидается список, получен {type(table_data).__name__}")
        if not table_data:
            return False
        if not all(isinstance(row, list) for row in table_data):
            raise TypeError("Все элементы table_data должны быть списками")
        if not all(isinstance(cell, str) for row in table_data for cell in row):
            raise ValueError("Все элементы таблицы должны быть строками")

        # Преобразование таблицы в текстовый формат для проверки
        table_text = "\n".join("|".join(cell.strip() for cell in row) 
                              for row in table_data if any(cell.strip() for cell in row))

        # Основные проверки таблицы
        has_valid_structure = (
            len(table_data) >= 2 and                      # Минимум 2 строки
            all(len(row) >= 2 for row in table_data) and  # Минимум 2 столбца в каждой строке
            not any(pattern.search(table_text)             # Отсутствие не-табличных паттернов
                for pattern in self.NON_TABLE_PATTERNS)
        )
        return has_valid_structure

    def extract_table_name(self, page, table_bbox, search_height=20):
        """
        Извлекает название таблицы из области над таблицей на странице PDF.
    
        Параметры:
            page (pdfplumber.Page): Страница PDF, на которой находится таблица
            table_bbox (tuple): Координаты таблицы (x0, top, x1, bottom)
            search_height (int): Высота области поиска над таблицей (в пунктах)
    
        Возвращает:
            str: Название таблицы или пустую строку, если название не найдено
        """
        try:
            # Получаем координаты верхней границы таблицы
            top = table_bbox[1]

            search_height = min(20, top - page.bbox[1])  # Не больше чем доступное пространство
            if search_height > 0:
                # Определяем область для поиска названия таблицы:
                title_area = (0, top - search_height, page.width, top)
                # Извлекаем текст из заданной области
                title_text = page.crop(title_area).extract_text()
            else:
                title_text = ""
            
            # Если текст не найден, возвращаем пустую строку
            if not title_text:
                return ""
    
            # Ищем шаблон "Табл. <номер>. <название>"
            match = re.search(r"^Табл\.\s*\d+\.\s*(.+)", title_text.strip())
            # Возвращаем название или весь текст, если шаблон не найден
            return match.group(1).strip() if match else ""
        except Exception as e:
            self.logger.error(
                f"Ошибка при извлечении наименования таблицы: {str(e)}",
                exc_info=True
            )
        return ""    

    def _extract_tables(self, page: 'pdfplumber.Page') -> List[Dict[str, Any]]:
        """Извлекает таблицы из страницы в структурированном виде.
    
        Args:
            page: Объект страницы PDF из библиотеки pdfplumber, из которой происходит извлечение таблиц.
        
        Returns:
            Список словарей, каждый из которых представляет структурированную таблицу:
            - name: Название таблицы (str)
            - headers: Заголовки столбцов (List[str])
            - rows: Данные таблицы (List[List[str]])
            - page: Номер страницы (int)
            - bbox: Координаты таблицы (Tuple[float, float, float, float])
        """
        tables = []
    
        # Получаем список объектов таблиц
        table_objects = page.find_tables(table_settings=self.settings)
    
        for table_obj in table_objects:
            try:
                # Получаем координаты таблицы ДО извлечения данных
                table_bbox = table_obj.bbox
        
                # Извлекаем данные таблицы
                table_data = table_obj.extract()
        
                # Очищаем данные (заменяем None на пустые строки)
                cleaned_table = [
                    [cell.strip() if cell and isinstance(cell, str) else "" 
                    for cell in row]
                    for row in table_data
                ]
        
                # Пропускаем пустые таблицы
                if not cleaned_table:
                    continue
            
                # Извлекаем название таблицы
                table_name = self.extract_table_name(page, table_bbox)
        
                # Проверяем валидность таблицы
                if not self.is_real_table(cleaned_table) or not table_name:
                    continue
        
                # Формируем структуру таблицы
                headers = cleaned_table[0]
                rows = cleaned_table[1:]
        
                tables.append({
                    "name": table_name,
                    "headers": headers,
                    "rows": rows,
                    "page": page.page_number,
                    "bbox": table_bbox  # Сохраняем координаты
                })
            except Exception as e:
                self.logger.error(
                    f"Ошибка обработки таблицы: {e}",
                    exc_info=True
                )

        if tables:
            self.doc_tables.extend(tables)

        return tables
    
    def check_bbox_overlap(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float],
        y_tolerance: int = 2
    ) -> bool:
        """Проверяет пересечение двух bounding box'ов по вертикали с заданным допуском.
    
        Args:
            bbox1: Координаты первого bounding box в формате (x0, y0, x1, y1)
            bbox2: Координаты второго bounding box в формате (x0, y0, x1, y1)
            y_tolerance: Допустимое отклонение по оси Y (по умолчанию 2)
        
        Returns:
            True если bounding box'ы пересекаются по вертикали (с учетом допуска),
            False в противном случае.
        
        Note:
            Координаты bounding box'ов ожидаются в порядке:
            (левая_граница, верхняя_граница, правая_граница, нижняя_граница)
        """
        # Проверяем пересечение по вертикали
        if (bbox1[1] >= bbox2[1] - y_tolerance) and (bbox1[3] <= bbox2[3] + y_tolerance):
            return True
        return False       

    def is_line_in_table(self, line, page_tables, y_tolerance=2):
        """
        Проверяет, находится ли строка внутри bbox какой-либо таблицы на странице.
    
        Args:
            line: dict с данными строки (должен содержать 'top' и 'bottom')
            page_tables: list таблиц страницы (каждая таблица должна иметь 'bbox')
            y_tolerance: допустимое отклонение по оси Y (чтобы учесть погрешности)
    
        Returns:
            bool: True если строка пересекается с таблицей, False иначе
        """
        line_top = line['top']
        line_bottom = line['bottom']
  
        for table in page_tables:
            table_bbox = table['bbox']  # (x0, top, x1, bottom)
            table_top = table_bbox[1] - y_tolerance
            table_bottom = table_bbox[3] + y_tolerance
        
            # Проверяем пересечение по вертикали
            if (line_top <= table_bottom) and (line_bottom >= table_top):
                return True
            
        return False

    def get_section_tables(self, section_bboxes, start_page, end_page):
        """
        Возвращает таблицы, принадлежащие текущей секции на основе:
            - пересечения bounding box с секцией на соответствующих страницах
            - нахождения на страницах секции (между start_page и end_page)

        Args:
            section_bboxes: list[dict] - список bbox'ов секции в формате 
                [{"page": int, "bbox": [x0, y0, x1, y1]}, ...]
            start_page: int - начальная страница секции
            end_page: int - конечная страница секции

        Returns:
            list: таблицы, принадлежащие секции
        """
        section_tables = []
    
        # Создаем словарь {page: bbox} для быстрого доступа
        page_bbox_map = {sb["page"]: sb["bbox"] for sb in section_bboxes}
    
        for table in self.doc_tables:
            table_page = table['page']
            table_bbox = table['bbox']
        
            # Проверяем, что таблица в диапазоне страниц секции
            if start_page <= table_page <= end_page:
                # Если для этой страницы есть bbox секции - проверяем пересечение
                if table_page in page_bbox_map:
                    if self.check_bbox_overlap(table_bbox, page_bbox_map[table_page]):
                        section_tables.append(table)
                else:
                    # Если bbox для страницы не указан, включаем таблицу без проверки пересечения
                    section_tables.append(table)
            
        return section_tables

    def is_toc_page(
        self,
        text: str,
        page_num: int,
        total_pages: int,
        check_length: int = 500
    ) -> bool:
        """Определяет, является ли страница частью оглавления/содержания.
    
        Args:
            text: Полный текст страницы для анализа
            page_num: Номер текущей страницы (начиная с 1)
            total_pages: Общее количество страниц в документе
            check_length: Количество символов от начала текста для проверки

        Returns:
            bool: True если страница идентифицирована как оглавление, 
                  False в противном случае или при ошибке
        """
        try:
            # Быстрая проверка без анализа
            if not text or not text.strip():
                return False
            
            if page_num < 1 or total_pages < 1 or page_num > total_pages:
                self.logger.warning(
                    f"Некорректные номера страниц: {page_num}/{total_pages}"
                )
                return False

            # Основная логика проверки
            if not (page_num <= 3 or page_num >= total_pages - 2):
                return False

            # Безопасное извлечение текста для анализа
            start_text = text[:min(check_length, len(text))].lower()
            line_count = len(text.splitlines())
        
            # Проверка паттернов
            has_toc_patterns = any(
                pattern.search(start_text) 
                for pattern in self.TOC_PATTERNS
                if hasattr(pattern, 'search')
            )
            return has_toc_patterns and line_count > 10
            
        except Exception as e:
            self.logger.error(
                f"Ошибка в is_toc_page (стр. {page_num}/{total_pages}): {str(e)}",
                exc_info=True
            )
            return False

    def get_language_doc(self, pdf_obj: pdfplumber.PDF) -> str:
        text = pdf_obj.pages[1].extract_text()[:500]  # Анализируем первые 500 символов
        return detect(text) if text else "unknown"

    def extract_document_type(self, pdf_obj: pdfplumber.PDF) -> Optional[str]:
        """
        Извлекает тип/назначение документа из первой страницы объекта PDF.
        
        Параметры:
            pdf_obj (pdfplumber.PDF): Открытый объект PDF из pdfplumber
            
        Возвращает:
            Optional[str]: Найденное назначение документа или None
        """
        if not len(pdf_obj.pages):
            return None

        try:
            first_page = pdf_obj.pages[0]
            text = first_page.extract_text() or ""
            
            # Поиск по скомпилированным паттернам
            for pattern in self.DOC_TYPE_PATTERNS:
                match = pattern.search(text)
                if match:
                    return match.group(0).strip()
            
            # Альтернативный поиск по структуре
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if len(lines) >= 3 and self.GUIDE_PATTERN.search(lines[2]):
                return lines[2]
                
        except Exception as e:
            self.logger.error(
                f"Ошибка при определении типа документа: {e}",
                exc_info=True
            )
        return None

    def extract_annotation(
        self,
        pdf_obj: pdfplumber.PDF,
        x_tolerance: int = 2,
        y_tolerance: int = 2,
    ) -> Optional[str]:
        """
        Извлекает аннотацию со второй страницы PDF.
    
        Параметры:
            pdf_obj: Открытый объект PDF из pdfplumber.
            x_tolerance, y_tolerance: Параметры для extract_text_lines.
            
        Возвращает:
            Текст аннотации или None, если не найдена.
        """
        if len(pdf_obj.pages) < 2:
            return None

        try:
            page = pdf_obj.pages[1]
            lines = page.extract_text_lines(
                x_tolerance=x_tolerance,
                y_tolerance=y_tolerance,
                keep_blank_chars=False,
            )
        
            annotation_lines = []
            found_annotation = False
        
            for line in lines:
                line_text = self.clean_pdf_text(line["text"])
            
                # Пропускаем колонтитулы
                if (line["top"] < self.default_top_margin 
                    or line["top"] > page.height - self.default_bottom_margin):
                    continue
            
                # Ищем начало аннотации
                if not found_annotation:
                    if self.ANNOTATION_PATTERN.search(line_text):
                        found_annotation = True
                    continue
            
                # Собираем все строки после нахождения паттерна
                annotation_lines.append(line_text.strip())
        
            annotation_text = " ".join(annotation_lines).strip()

            # Заменяет множественные пробелы на один
            annotation_text = re.sub(r'\s+', ' ', annotation_text)    
            # Убирает пробелы перед точками/запятыми
            annotation_text = re.sub(r'\s([.,])', r'\1', annotation_text) 
            # Обрабатывает пробелы у дефисов "-", и длинных тире
            annotation_text = re.sub(r'[—\-]\s+(\w)', r'\1', annotation_text)   
            return annotation_text if annotation_text else None
        
        except Exception as e:
            self.logger.error(
                f"Ошибка при извлечении аннотации со страницы 2: {e}",
                exc_info=True,
            )
            return None

    def convert_pdf_date(self, pdf_date_str: str) -> str:
        """
        Конвертирует PDF-дату (формат "D:YYYYMMDDHHMMSS...") в строку ISO 8601.
    
        Args:
            pdf_date_str: Дата в PDF-формате (начинается с "D:")
        
        Returns:
            Строка с датой в формате ISO 8601 (например, "2022-06-22T12:17:10+03:00")
        
        Raises:
            ValueError: Если формат даты некорректен
        """
        if not pdf_date_str.startswith("D:"):
            raise ValueError("Дата должна начинаться с 'D:'")

        date_part = pdf_date_str[2:]
        try:
            # Основная часть даты (YYYYMMDDHHMMSS)
            year = int(date_part[:4])
            month = int(date_part[4:6])
            day = int(date_part[6:8])
            hour = int(date_part[8:10])
            minute = int(date_part[10:12])
            second = int(date_part[12:14]) if len(date_part) >= 14 else 0
    
            # Временная зона
            tz_part = date_part[14:] if len(date_part) > 14 else ""
            tz_info = None
    
            if tz_part.startswith(("+", "-")):
                tz_sign = -1 if tz_part[0] == "-" else 1
                tz_hour = int(tz_part[1:3]) if len(tz_part) >= 3 else 0
                tz_minute = int(tz_part[4:6]) if len(tz_part) >= 6 else 0
                tz_offset = tz_sign * (tz_hour * 3600 + tz_minute * 60)
                tz_info = timezone(timedelta(seconds=tz_offset))
            elif tz_part.startswith("Z"):
                tz_info = timezone.utc
    
            dt = datetime(year, month, day, hour, minute, second, tzinfo=tz_info)
            return dt.isoformat()  # Возвращаем строку в ISO формате
    
        except Exception as e:
            self.logger.error(
                f"Ошибка парсинга даты '{pdf_date_str}': {e}",
                exc_info=True,
            )
            return None    

    def load_data(self, pdf_path: Union[str, Path], max_empty_pages: int = 2) -> List[Document]:
        """
        Загружает PDF и корректно разбивает на разделы, сохраняя принадлежность текста.
       
        Args:
            pdf_path: Путь к PDF-файлу (строка или Path-объект)
            max_empty_pages: Максимальное количество пустых страниц подряд перед завершением секции

        Returns:
            List[Document]: Список документов, каждый представляет отдельный раздел PDF

        Raises:
            FileNotFoundError: Если PDF-файл не существует
            PDFSyntaxError: Если файл не является валидным PDF
            ValueError: Если возникла ошибка при обработке содержимого PDF

        Process:
            1. Открывает PDF и обрабатывает каждую страницу
            2. Пропускает страницы оглавления
            3. Извлекает текст и таблицы (если включено)
            4. Группирует контент по разделам на основе нумерации
            5. Создает структурированные документы с метаданными
         """
        self.logger.start_timer(f"Обработка документа: {pdf_path}")
        documents = []
        current_section_pages = []
        current_hierarchy = []
        current_section_text = ""
        empty_page_count = 0
        doc_tables = []
        current_section_bbox = []
        text_bbox = []

        def create_root_document(pdf):
            self.root_doc_id  = uuid.uuid4().hex 

            doc =  Document(
                id_ = self.root_doc_id,
                metadata={
                    "type": "root_doc",
                    "title": pdf.metadata.get("Title", ""),
                    "author": pdf.metadata.get("Author", ""),
                    "source": pdf_path,
                    "mod_date": self.convert_pdf_date(pdf.metadata.get("ModDate", "")),
                    "total_pages": len(pdf.pages),
                    "lang" : self.get_language_doc(pdf),
                    "document_type" : self.extract_document_type(pdf),
                    "annotation" : self.extract_annotation(pdf),
                }
            )
            # Удаление неиспользуемыъ полей
            for field in ['excluded_embed_metadata_keys',
                          'excluded_llm_metadata_keys',
                          'text_resource',
                          'image_resource',
                          'audio_resource',
                          'video_resource',
                          'relationships',
                          'embedding']:
                if hasattr(doc, field):
                    delattr(doc, field)
            documents.append(doc)

        def _finalize_section():
            nonlocal current_section_pages, current_hierarchy, current_section_text, text_bbox, current_section_bbox
            if not current_section_pages or not current_section_text.strip():
                return

            start_page = current_section_pages[0]
            end_page = current_section_pages[-1]

            if text_bbox:
                current_section_bbox.append(text_bbox)
                
            lines = current_section_text.strip().split('\n')
            section_title = lines[0] if lines else ""
            section_body = '\n'.join(lines[1:]) if len(lines) > 1 else ""
        
            doc =  Document(
                text=section_body,
                metadata={
                    "type": "section",
                    "section_id": uuid.uuid4().hex,
                    "hierarchy": current_hierarchy.copy(),
                    "level": len(current_hierarchy),
                    "start_page": start_page,
                    "end_page": end_page,
                    "tables": self.get_section_tables(current_section_bbox, start_page, end_page),
                    "bbox": current_section_bbox,
                    "title": section_title,
                }
            )

            # Удаление неиспользуемыъ полей
            for field in ['excluded_embed_metadata_keys',
                          'excluded_llm_metadata_keys',
                          'image_resource',
                          'audio_resource',
                          'video_resource',
                          'relationships',
                          'embedding']:
                if hasattr(doc, field):
                    delattr(doc, field)
            
            documents.append(doc)
            text_bbox = []
            current_section_bbox = []
            current_section_pages = []
            current_section_text = ""
            current_hierarchy = []
    
        with pdfplumber.open(pdf_path) as pdf:
            create_root_document(pdf)
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text and self.is_toc_page(page_text, page_num, total_pages):
                        continue
                        
                    current_section_page = page_num
                   
                    if text_bbox:
                        current_section_bbox.append(text_bbox)
                        text_bbox = {
                            "page": page_num,
                            "bbox": [0, self.default_top_margin, page.width, 0]
                        }  
                    
                    # Извлекаем таблицы
                    tables = self._extract_tables(page) if self.include_tables else []
                                
                    lines = page.extract_text_lines(
                        x_tolerance=2,
                        y_tolerance=2,
                        keep_blank_chars=False
                    )

                    for line in lines:
                        line_text = self.clean_pdf_text(line['text'])
                        line_bbox = (line['x0'], line['top'], line['x1'], line['bottom'])

                        # Колонтитулы не обрабатываем
                        if line['top'] < self.default_top_margin or line['top'] > page.height - self.default_bottom_margin:
                            continue
                        
                        if self.include_tables and self.is_line_in_table(line, tables):
                            text_bbox["bbox"][3] = line['bottom']
                        else:
                            if not page_text:
                                empty_page_count += 1
                                if empty_page_count >= max_empty_pages:
                                    _finalize_section()
                                continue
        
                            empty_page_count = 0
                            
                            section_match = re.match(r'^(\d+(?:\.\d+)*)\.[^\S\r\n]+[A-ZА-Я]', line_text)
                            
                            if section_match:
                                # Финализируем текущую секцию перед началом новой
                                if current_section_text.strip():
                                    _finalize_section()
                    
                                # Начинаем новую секцию
                                current_hierarchy = section_match.group(1).split('.')
                                current_section_pages = [page_num]
                                
                                if line_text:
                                    current_section_text = line_text + "\n"
                                   
                                
                                text_bbox = {
                                    "page": page_num,
                                    "bbox": list(line_bbox)}    
                            else:
                                if current_section_text:
                                    text_bbox["bbox"][3] = line['bottom']
                                    current_section_pages.append(page_num)
                                    if line_text:
                                        current_section_text += line_text + "\n"
                                        
                                    
                
                except Exception as e:
                    print(f"Ошибка на странице {page_num}: {e}")
                    continue

            _finalize_section()  # Финализируем последнюю секцию
            self.logger.end_timer(f"Обработка документа: {pdf_path}")
        return documents