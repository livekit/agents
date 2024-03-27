"""
Converts the change Payload into native Python types.
"""
import json
from dateutil.parser import parse


abstime = "abstime"
_bool = "bool"  # bool is a keyword in python
date = "date"
daterange = "daterange"
float4 = "float4"
float8 = "float8"
int2 = "int2"
int4 = "int4"
int4range = "int4range"
int8 = "int8"
int8range = "int8range"
_json = "json"  # Potentially a library
jsonb = "jsonb"
money = "money"
numeric = "numeric"
oid = "oid"
reltime = "reltime"
time = "time"
timestamp = "timestamp"
timestamptz = "timestamptz"
timetz = "timetz"
tsrange = "tsrange"
tstzrange = "tstzrange"


"""
Takes an array of columns and an object of string values then converts each string value
to its mapped type.

:param columns:
:param records:
:param options: The map of various options that can be applied to the mapper

"""


def convert_change_data(columns, records, options={}):
    skip_types = options.get("skip_types") if options.get(
        "skip_types") == "undefined" else []
    return {
        key: convert_column(key, columns, records, skip_types)
        for key in records.keys()
    }


def convert_column(column_name, columns, records, skip_types):
    column = next(filter(lambda x: x.get("name") == column_name, columns))
    if not column or column.get("type") in skip_types:
        return noop(records[column_name])
    else:
        return convert_cell(column.get("type"), records[column_name])


def convert_cell(_type: str, string_value: str):
    try:
        if string_value is None:
            return None
        # If data type is an array
        if _type[0] == "_":
            array_value = _type[1:len(_type)]
            return to_array(string_value, array_value)
        # If it's not null then we need to convert it to the correct type
        if _type == abstime:
            return noop(string_value)
        elif _type == _bool:
            print("converted to bool")
            return to_boolean(string_value)
        elif _type == date:
            return noop(string_value)
        elif _type == daterange:
            return to_date_range(string_value)
        elif _type == float4:
            return to_float(string_value)
        elif _type == float8:
            return to_float(string_value)
        elif _type == int2:
            return to_int(string_value)
        elif _type == int4:
            return to_int(string_value)
        elif _type == int4range:
            return to_int_range(string_value)
        elif _type == int8:
            return to_int(string_value)
        elif _type == int8range:
            return to_int_range(string_value)
        elif _type == _json:
            return to_json(string_value)
        elif _type == jsonb:
            return to_json(string_value)
        elif _type == money:
            return to_float(string_value)
        elif _type == numeric:
            return to_float(string_value)
        elif _type == oid:
            return to_int(string_value)
        elif _type == reltime:
            # To allow users to cast it based on Timezone
            return noop(string_value)
        elif _type == time:
            # To allow users to cast it based on Timezone
            return noop(string_value)
        elif _type == timestamp:
            return to_timestamp_string(
                string_value
            )  # Format to be consistent with PostgREST
        elif _type == timestamptz:
            return parse(string_value)
        elif _type == timetz:
            return parse(string_value)
        elif _type == tsrange:
            return to_date_range(string_value)
        elif _type == tstzrange:
            return to_date_range(string_value)
        else:
            # All the rest will be returned as strings
            return noop(string_value)

    except Exception as e:
        print(
            f"Could not convert cell of type {_type} and value {string_value}")
        print(f"This is the error {e}")
        return string_value


def noop(string_value: str):
    return string_value


def to_boolean(string_value: str):
    if string_value == "t":
        return True
    elif string_value == "f":
        return False
    else:
        return None


def to_date(string_value: str):
    return parse(string_value)


def to_date_range(string_value: str):
    arr = json.dumps(string_value)
    return [parse(arr[0]), parse(arr[1])]


def to_float(string_value):
    return float(string_value)


def to_int(string_value: str):
    return int(string_value)


def to_int_range(string_value: str):
    arr = json.loads(string_value)
    return [int(arr[0]), int(arr[1])]


def to_json(string_value: str):
    return json.loads(string_value)


"""
Converts a Postgres array into a native python list.
>>> to_array('{1,2,3,4}', 'int4')
    [1,2,3,4]
>>> to_array('{}', 'int4')
    []
"""


def to_array(string_value: str, type: str):
    # this takes off the '{' & '}'
    string_enriched = string_value[1: len(string_value) - 1]

    # Converts the string into an array
    # if string is empty (meaning the array was empty), an empty array will be immediately returned
    string_array = string_enriched.split(
        ",") if len(string_enriched) > 0 else []
    return list(map(lambda string: convert_cell(type, string), string_array))


"""
Fixes timestamp to be ISO-8601. Swaps the space between the date and time for a 'T'
>>> to_timestamp_string('2019-09-10 00:00:00')
    '2019-09-10T00:00:00'
"""


def to_timestamp_string(string_value: str):
    return string_value.replace(" ", "T")
