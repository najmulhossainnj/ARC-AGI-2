from enum import Enum

class ValueType(str, Enum):
    GRID = "Grid"
    SCENE = "Scene"
    OBJECT = "Object"
    OBJECT_SET = "ObjectSet"
    COLOR = "Color"
    INTEGER = "Integer"
    DIRECTION = "Direction"
    AXIS = "Axis"
    BOOLEAN = "Boolean"
    ANY = "Any"
