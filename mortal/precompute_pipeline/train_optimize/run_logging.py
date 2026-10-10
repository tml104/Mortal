"""本次训练的 Python 文本双写；墙钟仅用于文件名和行前缀。"""

from datetime import datetime
import logging
import re
import sys
import threading
import time
import traceback
from zoneinfo import ZoneInfo


BEIJING = ZoneInfo('Asia/Shanghai')
LOG_NAME = re.compile(r'log-[0-9]{6}-[0-9]{6}\.log')


def _prefix(wall_clock):
    return datetime.fromtimestamp(wall_clock(), BEIJING).strftime('[%y%m%d-%H:%M:%S] ')


def exception_reported(error):
    """CLI 不再重复打印已被本次日志会话完整记录的异常。"""
    return getattr(error, '_mortal_log_reported', False)


def report_terminal_exception(error, stream=None, wall_clock=time.time, *, chain=True):
    """文件不可用时直接写原终端，避免重新进入故障日志流。"""
    stream = sys.stderr if stream is None else stream
    try:
        for part in traceback.format_exception(type(error), error, error.__traceback__, chain=chain):
            for line in part.splitlines():
                stream.write(_prefix(wall_clock) + line + '\n')
                stream.flush()
    except Exception:
        # 终端本身也可能不可写；调用方仍须传播原始失败。
        pass
    error._mortal_log_reported = True


def is_run_log(path):
    """清理仅按普通日志文件的约定名称保留，不读取历史内容。"""
    return LOG_NAME.fullmatch(path.name) is not None and path.is_file()


class _TimestampedStream:
    """同一输出流的分段写入共用物理行前缀。"""

    def __init__(self, session, original):
        self.session, self.original = session, original

    def write(self, text):
        return self.session.write(self, text)

    def flush(self):
        self.session.flush(self)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def isatty(self):
        return self.original.isatty()

    def fileno(self):
        return self.original.fileno()

    @property
    def encoding(self):
        return self.original.encoding

    @property
    def errors(self):
        return self.original.errors


class TrainingLog:
    """按行同步写入文件与原终端；本次会话独立于 checkpoint。"""

    def __init__(self, output_dir, started_at, wall_clock):
        self.wall_clock = wall_clock
        stamp = datetime.fromtimestamp(started_at, BEIJING).strftime('%y%m%d-%H%M%S')
        self.path = output_dir / f'log-{stamp}.log'
        self.lock = threading.RLock()
        self.active = None
        self.file = None
        self.failure = None
        self.handlers = []

    def __enter__(self):
        self.original_out, self.original_err = sys.stdout, sys.stderr
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open('a+b')
            # 同秒追加时，先结束上次异常终止可能留下的半行。
            if self.file.tell():
                self.file.seek(-1, 2)
                if self.file.read(1) != b'\n':
                    self.file.write(b'\n')
                    self.file.flush()
        except Exception as error:
            self._close_file()
            report_terminal_exception(error, self.original_err, self.wall_clock)
            raise
        self.stdout = _TimestampedStream(self, self.original_out)
        self.stderr = _TimestampedStream(self, self.original_err)
        sys.stdout, sys.stderr = self.stdout, self.stderr
        self.original_handle_error = logging.Handler.handleError
        def propagate_sink_error(handler, record):
            """标准 StreamHandler 不得吞掉本会话已确认的输出故障。"""
            if getattr(handler, 'stream', None) is self.stdout or getattr(handler, 'stream', None) is self.stderr:
                self.raise_if_failed()
            return self.original_handle_error(handler, record)
        self.handle_error = propagate_sink_error
        logging.Handler.handleError = self.handle_error
        try:
            self._capture_handlers()
        except BaseException:
            self._restore_outputs()
            raise
        return self

    def _stream_handlers(self):
        """遍历注册 logger 的输出 handler，包括运行中创建的对象。"""
        loggers = [logging.getLogger(), *list(logging.Logger.manager.loggerDict.values())]
        seen = set()
        for logger in loggers:
            for handler in getattr(logger, 'handlers', ()):
                if handler in seen or not isinstance(handler, logging.StreamHandler):
                    continue
                seen.add(handler)
                yield handler

    def _capture_handlers(self):
        """已绑定旧 stdout/stderr 的 Python logging handler 同样纳入会话。"""
        for handler in self._stream_handlers():
            handler.acquire()
            try:
                original = handler.stream
                if original is self.original_out or original is self.original_err:
                    self.handlers.append(handler)
                    handler.stream = self.stdout if original is self.original_out else self.stderr
            finally:
                handler.release()

    def _restore_handlers(self):
        """直接还原原输出流，避免 setStream 对故障流再次 flush。"""
        for handler in set(self.handlers).union(self._stream_handlers()):
            handler.acquire()
            try:
                if handler.stream is self.stdout or handler.stream is self.stderr:
                    handler.stream = handler.stream.original
            finally:
                handler.release()
        self.handlers.clear()

    def _restore_outputs(self):
        """还原会话内的全局输出绑定，再释放文件。"""
        try:
            self._restore_handlers()
        finally:
            if logging.Handler.handleError is self.handle_error:
                logging.Handler.handleError = self.original_handle_error
            sys.stdout, sys.stderr = self.original_out, self.original_err
            self._close_file()

    def _write_both(self, stream, text):
        """一次生成的前缀和文本同时用于文件及对应原输出流。"""
        self.raise_if_failed()
        try:
            encoded = text.encode('utf-8')
            if self.file.write(encoded) != len(encoded):
                raise OSError('日志文件未完整写入')
            self.file.flush()
            stream.original.write(text)
            stream.original.flush()
        except Exception as error:
            self.failure = error
            raise

    def raise_if_failed(self):
        """把其他 Python 线程的日志故障传回训练边界，不访问文件或设备。"""
        if self.failure is not None:
            raise self.failure

    def _finish_line(self):
        if self.active is not None:
            self._write_both(self.active, '\n')
            self.active = None

    def write(self, stream, text):
        """换流时结束未完成行，避免两路输出混成无前缀的一行。"""
        if not isinstance(text, str):
            raise TypeError('write() requires str')
        with self.lock:
            if self.active is not None and self.active is not stream:
                self._finish_line()
            pieces = text.split('\n')
            for index, piece in enumerate(pieces):
                newline = index < len(pieces) - 1
                if not piece and not newline:
                    continue
                prefix = ''
                if self.active is None:
                    prefix = _prefix(self.wall_clock)
                self._write_both(stream, prefix + piece + ('\n' if newline else ''))
                self.active = None if newline else stream
        return len(text)

    def flush(self, stream):
        """刷新不强制换行，后续片段仍属于同一物理行。"""
        with self.lock:
            self.raise_if_failed()
            try:
                self.file.flush()
                stream.original.flush()
            except Exception as error:
                self.failure = error
                raise

    def _close_file(self):
        """关闭错误不能覆盖更早的训练或写入失败。"""
        if self.file is not None:
            try:
                self.file.close()
            except Exception as error:
                if self.failure is None:
                    self.failure = error

    def __exit__(self, exc_type, exc, tb):
        """训练资源收尾已结束后记录异常，再恢复终端和关闭文件。"""
        try:
            with self.lock:
                if exc is not None and not (isinstance(exc, SystemExit) and exc.code in (None, 0)):
                    if self.failure is None:
                        try:
                            self._finish_line()
                            traceback.print_exception(exc_type, exc, tb, file=self.stderr)
                            exc._mortal_log_reported = True
                        except Exception:
                            report_terminal_exception(exc, self.original_err, self.wall_clock)
                    else:
                        report_terminal_exception(exc, self.original_err, self.wall_clock)
                if self.failure is None:
                    try:
                        self._finish_line()
                        self.file.flush()
                    except Exception as error:
                        self.failure = error
        finally:
            self._restore_outputs()
        if self.failure is not None:
            if self.failure is not exc:
                report_terminal_exception(self.failure, self.original_err, self.wall_clock, chain=exc is None)
            if exc is None or (isinstance(exc, SystemExit) and exc.code in (None, 0)):
                raise self.failure
