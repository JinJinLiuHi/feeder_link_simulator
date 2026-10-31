"""Filter one optional performance-counter message, preserving all runtime errors."""

import logging


class OptionalTritonCounter(logging.Filter):
    def filter(self, record):
        return not (
            record.name == "torch.utils.flop_counter"
            and record.levelno == logging.WARNING
            and record.getMessage()
            == "triton not found; flop counting will not work for triton kernels"
        )


def configure_runtime_logging():
    logger = logging.getLogger("torch.utils.flop_counter")
    if not any(isinstance(item, OptionalTritonCounter) for item in logger.filters):
        logger.addFilter(OptionalTritonCounter())
