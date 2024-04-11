import logging


class ExtraLogFormatter(logging.Formatter):
    def __init__(self, formatter):
        super().__init__()
        self._formatter = formatter

    def format(self, record):
        # less hacky solution?
        dummy = logging.LogRecord("", 0, "", 0, None, None, None)
        extra_txt = "\t "
        for k, v in record.__dict__.items():
            if k not in dummy.__dict__:
                extra_txt += " {}={}".format(k, str(v).replace("\n", " "))
        message = self._formatter.format(record)
        return message + extra_txt


def setup_logging(log_level: str, production: bool = True) -> None:
    h = logging.StreamHandler()

    if production:
        ## production mode, json logs
        from pythonjsonlogger import jsonlogger

        class CustomJsonFormatter(jsonlogger.JsonFormatter):
            def add_fields(self, log_record, record, message_dict):
                super().add_fields(log_record, record, message_dict)
                if "taskName" in log_record:
                    log_record.pop("taskName")
                log_record["level"] = record.levelname

        f = CustomJsonFormatter("%(asctime)s %(level)s %(name)s %(message)s")
        h.setFormatter(f)
    else:
        ## dev mode, colored logs & show all extra
        import colorlog

        f = ExtraLogFormatter(
            colorlog.ColoredFormatter(
                "%(asctime)s %(log_color)s%(levelname)-4s %(bold_white)s %(name)s %(reset)s %(message)s",
                log_colors={
                    **colorlog.default_log_colors,
                    "DEBUG": "blue",
                },
            )
        )

        h.setFormatter(f)

    logging.root.addHandler(h)
    logging.root.setLevel(log_level)
