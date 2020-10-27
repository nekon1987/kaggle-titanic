

def Variated(number_of_variations: int):
    def func_wrapper(func):
        def final_wrapper(*args, **kwargs):
            final_wrapper.number_of_variations = number_of_variations
            return func(*args, **kwargs)
        return final_wrapper
    return func_wrapper