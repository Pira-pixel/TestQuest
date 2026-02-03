#!/usr/bin/env python3
"""
Программа для тестирования производительности веб-серверов
"""

import argparse
import asyncio
import re
import sys
import json
import time
import os
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse
import aiohttp
import aiofiles


class RequestStats:
    """Класс для хранения статистики запросов"""

    def __init__(self, host: str):
        self.host = host
        self.success = 0
        self.failed = 0
        self.errors = 0
        self.times: List[float] = []

    def add_result(self, status_code: Optional[int], duration: float):
        """Добавить результат запроса в статистику"""
        if status_code is None:
            self.errors += 1
        elif 200 <= status_code < 400:
            self.success += 1
            self.times.append(duration)
        else:
            self.failed += 1

    @property
    def min_time(self) -> float:
        """Минимальное время выполнения"""
        return min(self.times) if self.times else 0.0

    @property
    def max_time(self) -> float:
        """Максимальное время выполнения"""
        return max(self.times) if self.times else 0.0

    @property
    def avg_time(self) -> float:
        """Среднее время выполнения"""
        return sum(self.times) / len(self.times) if self.times else 0.0

    def to_dict(self) -> Dict:
        """Преобразовать статистику в словарь"""
        return {
            "host": self.host,
            "success": self.success,
            "failed": self.failed,
            "errors": self.errors,
            "min": round(self.min_time, 4),
            "max": round(self.max_time, 4),
            "avg": round(self.avg_time, 4) if self.times else 0.0
        }


class URLValidator:
    """Класс для валидации URL"""

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Проверить валидность URL"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    @staticmethod
    def normalize_url(url: str) -> str:
        """Нормализовать URL (добавить http:// если нет схемы)"""
        if not url.startswith(('http://', 'https://')):
            return f'http://{url}'
        return url


class Benchmark:
    """Класс для выполнения нагрузочного тестирования"""

    def __init__(self, count: int = 1):
        self.count = count
        self.stats: Dict[str, RequestStats] = {}
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_host(self, host: str) -> RequestStats:
        """Протестировать один хост"""
        stats = RequestStats(host)

        for i in range(self.count):
            start_time = time.time()
            try:
                async with self.session.get(host) as response:
                    duration = time.time() - start_time
                    stats.add_result(response.status, duration)

            except aiohttp.ClientError as e:
                duration = time.time() - start_time
                stats.add_result(None, duration)
            except Exception as e:
                duration = time.time() - start_time
                stats.add_result(None, duration)

        return stats

    async def run(self, hosts: List[str]) -> Dict[str, RequestStats]:
        """Запустить тестирование для всех хостов"""
        tasks = [self.test_host(host) for host in hosts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue
            self.stats[hosts[i]] = result

        return self.stats


class OutputFormatter:
    """Класс для форматирования вывода результатов"""

    @staticmethod
    def format_stats(stats_dict: Dict[str, RequestStats]) -> str:
        """Отформатировать статистику в читаемый вид"""
        output_lines = []

        for host, stats in stats_dict.items():
            output_lines.append(f"{'=' * 60}")
            output_lines.append(f"Host: {stats.host}")
            output_lines.append(f"{'-' * 60}")
            output_lines.append(f"Success:        {stats.success}")
            output_lines.append(f"Failed:         {stats.failed}")
            output_lines.append(f"Errors:         {stats.errors}")
            output_lines.append(f"Min time:       {stats.min_time:>8.4f} сек")
            output_lines.append(f"Max time:       {stats.max_time:>8.4f} сек")
            output_lines.append(f"Avg time:       {stats.avg_time:>8.4f} сек")
            output_lines.append(f"{'=' * 60}\n")

        return "\n".join(output_lines)

    @staticmethod
    def format_json(stats_dict: Dict[str, RequestStats]) -> str:
        """Отформатировать статистику в JSON"""
        data = {host: stats.to_dict() for host, stats in stats_dict.items()}
        return json.dumps(data, indent=2, ensure_ascii=False)

    @staticmethod
    def format_csv(stats_dict: Dict[str, RequestStats]) -> str:
        """Отформатировать статистику в CSV"""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(['Host', 'Success', 'Failed', 'Errors', 'Min', 'Max', 'Avg'])
        for stats in stats_dict.values():
            writer.writerow([
                stats.host,
                stats.success,
                stats.failed,
                stats.errors,
                f"{stats.min_time:.4f}",
                f"{stats.max_time:.4f}",
                f"{stats.avg_time:.4f}"
            ])

        return output.getvalue()


def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Программа для нагрузочного тестирования веб-серверов\n'
                    'Пример: python bench.py -H https://ya.ru,https://google.com -C 5',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    host_source = parser.add_mutually_exclusive_group(required=True)
    host_source.add_argument(
        '-H', '--hosts',
        type=str,
        help='Список хостов через запятую без пробелов (пример: https://ya.ru,https://google.com)'
    )
    host_source.add_argument(
        '-F', '--file',
        type=str,
        help='Путь к файлу со списком адресов (по одному на строку)'
    )

    parser.add_argument(
        '-C', '--count',
        type=int,
        default=1,
        help='Количество запросов на каждый хост (по умолчанию: 1)'
    )

    parser.add_argument(
        '-O', '--output',
        type=str,
        help='Путь к файлу для сохранения результатов'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'json', 'csv'],
        default='text',
        help='Формат вывода результатов (по умолчанию: text)'
    )

    return parser.parse_args()


async def read_hosts_from_file(filepath: str) -> List[str]:
    """Прочитать список хостов из файла"""
    hosts = []
    # Если путь относительный, делаем его абсолютным относительно текущей папки
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.getcwd(), filepath)

    async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
        async for line in f:
            url = line.strip()
            if url and not url.startswith('#'):
                hosts.append(url)
    return hosts


async def main():
    """Основная функция программы"""
    args = parse_arguments()

    if args.count <= 0:
        print("Ошибка: параметр --count должен быть положительным числом")
        sys.exit(1)

    # Получение списка хостов
    try:
        if args.hosts:
            hosts = [h.strip() for h in args.hosts.split(',')]
        else:
            hosts = await read_hosts_from_file(args.file)
    except FileNotFoundError:
        print(f"Ошибка: файл '{args.file}' не найден")
        print(f"Текущая папка: {os.getcwd()}")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при чтении хостов: {e}")
        sys.exit(1)

    # Валидация хостов
    valid_hosts = []
    invalid_hosts = []

    for host in hosts:
        normalized = URLValidator.normalize_url(host)
        if URLValidator.is_valid_url(normalized):
            valid_hosts.append(normalized)
        else:
            invalid_hosts.append(host)

    if invalid_hosts:
        print("Предупреждение: невалидные хосты будут проигнорированы:")
        for host in invalid_hosts:
            print(f"  - {host}")
        print()

    if not valid_hosts:
        print("Ошибка: нет валидных хостов для тестирования")
        sys.exit(1)

    print(f"Начинаю тестирование {len(valid_hosts)} хостов...")
    print(f"Количество запросов на хост: {args.count}")
    print("-" * 60)

    # Выполнение тестирования
    try:
        async with Benchmark(count=args.count) as benchmark:
            stats = await benchmark.run(valid_hosts)

        # Форматирование результатов
        if args.format == 'json':
            output = OutputFormatter.format_json(stats)
        elif args.format == 'csv':
            output = OutputFormatter.format_csv(stats)
        else:
            output = OutputFormatter.format_stats(stats)

        # Вывод или сохранение результатов
        if args.output:
            try:
                # Если путь относительный, сохраняем в текущей папке
                if not os.path.isabs(args.output):
                    args.output = os.path.join(os.getcwd(), args.output)

                async with aiofiles.open(args.output, 'w', encoding='utf-8') as f:
                    await f.write(output)
                print(f"\n✓ Результаты сохранены в файл: {args.output}")
            except Exception as e:
                print(f"\n✗ Ошибка при сохранении: {e}")
                print("\nРезультаты:")
                print(output)
        else:
            print("\n" + "=" * 60)
            print("ИТОГОВАЯ СТАТИСТИКА")
            print("=" * 60 + "\n")
            print(output)

    except KeyboardInterrupt:
        print("\n\n✗ Тестирование прервано")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        sys.exit(1)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограмма завершена")
        sys.exit(0)
