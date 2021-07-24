##^# ops import and utils ######################################################
import os, pickle, time as time_module, pdb, math
from pprint import pprint

import torch, numpy as np
from torch.utils.tensorboard import SummaryWriter


##$#############################################################################
##^# torch utils ###############################################################
def topts(A):
    return dict(device=A.device, dtype=A.dtype)


ss = lambda x, dim=(): torch.sum(x ** 2, dim=dim)
t = lambda x: x.transpose(-1, -2)
diag = lambda x: x.diagonal(0, -1, -2)
vec = lambda x: x.reshape(-1)
identity = lambda x: x
is_equal = (
    lambda a, b: (type(a) == type(b))
    and (a.shape == b.shape)
    and (torch.norm(a - b) / math.sqrt(a.numel()) < 1e-7)
)


def normalize(x, dim=-2, params=None, min_std=1e-3):
    if params is None:
        x_mu = torch.mean(x, dim, keepdim=True)
        x_std = torch.maximum(
            torch.std(x, dim, keepdim=True), torch.tensor(min_std, **topts(x))
        )
    else:
        x_mu, x_std = params
    return (x - x_mu) / x_std, (x_mu, x_std)


unnormalize = lambda x, params: x * params[1] + params[0]

onehot = lambda *args, **kwargs: torch.nn.functional.one_hot(
    args[0].to(int), *args[1:], **kwargs
).to(args[0].dtype)

t2n = (
    lambda x: np.copy(x.detach().cpu().clone().numpy().astype(np.float64))
    if isinstance(x, torch.Tensor)
    else x
)
n2t = lambda x, device=None, dtype=None: torch.as_tensor(
    x, device=device, dtype=dtype
)

##$#############################################################################
##^# timing ####################################################################
def elapsed(name, t1, end=None):
    t2 = time_module.time()
    name = name if len(name) <= 20 else name[:17] + "..."
    msg = "%20s took %9.4e ms" % (name, (t2 - t1) * 1e3)
    if end is not None:
        print(msg, end=end)
    else:
        print(msg)


time = time_module.time
##$#############################################################################
##^# table printing utility class ##############################################
class TablePrinter:
    def __init__(self, names, fmts=None, prefix="", use_writer=False):
        self.names = names
        self.fmts = fmts if fmts is not None else ["%9.4e" for _ in names]
        self.widths = [
            max(self.calc_width(fmt), len(name)) + 2
            for (fmt, name) in zip(fmts, names)
        ]
        self.prefix = prefix
        self.writer = None
        if use_writer:
            try:
                self.writer = SummaryWriter(flush_secs=1)
                self.iteration = 0
            except NameError:
                print("SummaryWriter not available, ignoring")

    def calc_width(self, fmt):
        f = fmt[-1]
        width = None
        if f == "f" or f == "e" or f == "d" or f == "i":
            width = max(len(fmt % 1), len(fmt % (-1)))
        elif f == "s":
            width = len(fmt % "")
        else:
            raise ValueError("I can't recognized the [%s] print format" % fmt)
        return width

    def pad_field(self, s, width, lj=True):
        # lj -> left justify
        assert len(s) <= width
        rem = width - len(s)
        if lj:
            return (" " * (rem // 2)) + s + (" " * ((rem // 2) + (rem % 2)))
        else:
            return (" " * ((rem // 2) + (rem % 2))) + s + (" " * (rem // 2))

    def make_row_sep(self):
        return "+" + "".join([("-" * width) + "+" for width in self.widths])

    def make_header(self):
        s = self.prefix + self.make_row_sep() + "\n"
        s += self.prefix
        for (name, width) in zip(self.names, self.widths):
            s += "|" + self.pad_field("%s" % name, width, lj=True)
        s += "|\n"
        return s + self.prefix + self.make_row_sep()

    def make_footer(self):
        return self.prefix + self.make_row_sep()

    def make_values(self, vals):
        assert len(vals) == len(self.fmts)
        s = self.prefix + ""
        for (val, fmt, width) in zip(vals, self.fmts, self.widths):
            s += "|" + self.pad_field(fmt % val, width, lj=False)
        s += "|"

        if self.writer is not None:
            for (name, val) in zip(self.names, vals):
                self.writer.add_scalar(name, val, self.iteration)
            self.iteration += 1

        return s

    def print_header(self):
        print(self.make_header())

    def print_footer(self):
        print(self.make_footer())

    def print_values(self, vals):
        print(self.make_values(vals))


##$#############################################################################
##^# solution caching decorator ################################################
def to_tuple_(arg):
    if isinstance(arg, np.ndarray):
        return arg.tobytes()
    elif isinstance(arg, torch.Tensor):
        return to_tuple_(arg.cpu().detach().numpy())
    else:
        return to_tuple_(np.array(arg))

def to_tuple(*args):
    return tuple(to_tuple_(arg) for arg in args)


def fn_with_sol_cache(fwd_fn, cache=None):
    def inner_decorator(fn):
        nonlocal cache
        cache = cache if cache is None else cache

        def fn_with_sol(*args, **kwargs):
            cache, sol_key = fn_with_sol.cache, to_tuple(*args)
            sol = fwd_fn(*args) if not sol_key in cache else cache[sol_key]
            cache.setdefault(sol_key, sol)
            return fn_with_sol.fn(sol, *args, **kwargs)

        fn_with_sol.cache = cache
        fn_with_sol.fn = fn
        return fn_with_sol

    return inner_decorator


##$#############################################################################
