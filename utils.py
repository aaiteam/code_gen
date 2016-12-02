class ExecutionResult:

    def __init__(self, output="", raised_exception=False, exception_type=None):
        self.output = output
        self.raised_exception = raised_exception
        self.exception_type = exception_type

    def __str__(self):
        if self.raised_exception:
            return "Raised exception of type {}, code output: {}".format(self.exception_type, self.output)
        else:
            return "No exception, code output: {}".format(self.output)
