"""Historical compatibility entry point. New CLI: control_q.run."""
from .control_q.config import *
from .control_q.metrics import *
from .control_q.models import *
from .control_q.diagnostics.terminal_value import *
from .control_q.reporting.figures import *
from .control_q.legacy_cli import *

if __name__ == "__main__":
    main()
