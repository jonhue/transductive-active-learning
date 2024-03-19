from typing import Union
from jaxtyping import Array, Bool, Float, Int

ScalarBoolArray = Bool[Array, ""]
ScalarBool = Union[bool, ScalarBoolArray]
ScalarIntArray = Int[Array, ""]
ScalarInt = Union[int, ScalarIntArray]
ScalarFloatArray = Float[Array, ""]
ScalarFloat = Union[float, ScalarFloatArray]

KeyArray = Array
