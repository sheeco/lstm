# coding:utf-8


def match(shape1, shape2):
    return (len(shape1) == len(shape2) and
            all(s1 is None or s2 is None or s1 == s2
                for s1, s2 in zip(shape1, shape2)))


def warn(info):
    if not isinstance(info, str):
        raise ValueError("warn @ config: \n\tType of arg 'info' must be <str> instead of <" + type(str) + ">")
    print "[Warning] " + info
