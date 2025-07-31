from typing import Callable
from ast import literal_eval

import xarray as xr
import libcst as cst
import inspect
import libcst.matchers as m

test_file = "/Users/u1166368/xarray/tos_Omon_CESM2-WACCM_historical_r2i1p1f1_gr_185001-201412.nc"

ds = xr.open_dataset(test_file, decode_times=False)


def sequential(ds):
    return ds.mean(dim="time").mean(dim="lat").mean(dim="lon")


def grouped(ds):
    return ds.mean(dim=["time", "lat", "lon"])


def extract_call_args(call_node: cst.Call) -> dict:
    """
    Take a cst Call Node and, assuming only kwargs, extract that into a dict
    """
    kwargs = {}
    for arg in call_node.args:
        if arg.keyword is None:
            raise TypeError("Only dealing with kwargs for now")
        else:
            key = arg.keyword.value
            try:
                kwargs[key] = literal_eval(arg.value.value)
            except AttributeError:  # Can't literal eval a list
                kwargs[key] = [
                    literal_eval(arg.value.value) for arg in arg.value.elements
                ]
            # ^ Arg.value.value looks stupid, but arg.value is a cst Node itself
            # For catalogs, it's usually a cst.SimpleString or something like that,
            # so we could probably literal eval it?

    return kwargs


def format_args(kwargs: dict) -> str:
    """
    Format args and kwargs into a string representation
    """
    if kwargs:
        kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        return kwargs_str
    return ""


def merge_args(pop_args: cst.Arg, keep_args: cst.Arg) -> cst.Arg:
    """
    Take arguments from pop_node, and merge them with keep_node.
    """
    keep_arg_keys = [arg.keyword.value for arg in keep_args if arg.keyword is not None]
    for arg in pop_args:
        if arg.keyword is None:
            raise TypeError("Only dealing with kwargs for now")
        if arg.keyword.value not in keep_arg_keys:
            raise TypeError("Not handling this for now")

    # Collect all elements, flattening any existing lists
    all_elements = []

    # Add elements from pop_args (the inner call)
    for arg in pop_args:
        if isinstance(arg.value, cst.List):
            # If it's already a list, extract its elements
            all_elements.extend(arg.value.elements)
        else:
            # If it's a single value, wrap it in an Element
            all_elements.append(cst.Element(value=arg.value))

    # Add elements from keep_args (the outer call)
    for arg in keep_args:
        if isinstance(arg.value, cst.List):
            # If it's already a list, extract its elements
            all_elements.extend(arg.value.elements)
        else:
            # If it's a single value, wrap it in an Element
            all_elements.append(cst.Element(value=arg.value))

    # Create a new Arg node with the flattened list
    flattened_list = cst.List(elements=all_elements)

    return cst.Arg(
        keyword=cst.Name(keep_arg_keys[0]),
        value=flattened_list,
    )


class ChainSimplifier(cst.CSTTransformer):
    """
    Transform chained calls by removing intermediate method calls
    Example: ds.search(...).search(...).to_dataset_dict()
    becomes: ds.to_dataset_dict()
    """

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        # Use matcher to identify the pattern: any_method(search_call(...))
        search_pattern = m.Call(
            func=m.Attribute(value=m.Call(func=m.Attribute(attr=m.Name("mean"))))
        )

        if m.matches(updated_node, search_pattern):
            # Extract the method name and inner call
            method_name = updated_node.func.attr.value
            inner_call = updated_node.func.value

            kwargs = extract_call_args(inner_call)

            # Replace the value with the inner call's value
            # This effectively removes the search() call
            new_args = merge_args(inner_call.args, updated_node.args)
            new_func = updated_node.func.with_changes(value=inner_call.func.value)
            return updated_node.with_changes(func=new_func, args=[new_args])

        return updated_node


def ast_transform(func: Callable) -> Callable:
    """
    Transform a function to remove chained calls. Computes the transformation
    on every invoation - not good.
    """

    def wrapper(*args, **kwargs):
        cst_for_mods = cst.parse_module(inspect.getsource(func))
        transformer = ChainSimplifier()
        transformed_cst = cst_for_mods.visit(transformer)
        exec(transformed_cst.code, globals(), locals())
        return func(*args, **kwargs)

    return wrapper


def ast_transform_fast(func: Callable) -> Callable:
    """
    Transform a function to remove chained calls.
    """
    # Get the source BEFORE creating the wrapper
    original_source = inspect.getsource(func)

    # Do the transformation ONCE when the decorator is applied
    cst_for_mods = cst.parse_module(original_source)
    transformer = ChainSimplifier()
    transformed_cst = cst_for_mods.visit(transformer)
    transformed_code = transformed_cst.code

    # Compile the transformed code once and execute it to get the function
    compiled_code = compile(transformed_code, "<transformed>", "exec")
    local_vars = {}
    exec(compiled_code, globals(), local_vars)

    # Get the transformed function once during decoration
    func_name = func.__name__
    transformed_func = local_vars[func_name]

    def wrapper(*args, **kwargs):
        # Simply call the pre-compiled transformed function
        return transformed_func(*args, **kwargs)

    # Store the transformed code for inspection
    wrapper.transformed_code = transformed_code
    return wrapper


@ast_transform_fast
def sequential_transformed_fast(ds):
    return ds.mean(dim="time").mean(dim="lat").mean(dim="lon")


@ast_transform
def sequential_transformed(ds):
    """
    Sequentially apply mean over dimensions time, lat, lon.
    """
    return ds.mean(dim="time").mean(dim="lat").mean(dim="lon")


# Now time them

import timeit

print("Timing original sequential function:")
print(timeit.timeit(lambda: sequential(ds), number=5))
print("Timing transformed sequential function:")
print(timeit.timeit(lambda: sequential_transformed(ds), number=5))
print("Timing transformed fast sequential function:")
print(timeit.timeit(lambda: sequential_transformed_fast(ds), number=5))
