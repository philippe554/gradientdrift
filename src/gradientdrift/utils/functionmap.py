
import inspect
import jax

def getFunctionMap():
    namespaces = [jax.numpy, jax.scipy, jax.nn, jax.random]
    defaultNamespaces = ["numpy", "scipy", "stats", "nn", "random"]
    functionMap = {}

    def recursiveSearchForFunctions(ns):
        nsName = ns.__name__
        if nsName.startswith('jax.'):
            nsName = nsName[4:]

        nsNameShort = nsName
        for defaultNs in defaultNamespaces:
            if nsNameShort.startswith(defaultNs):
                nsNameShort = nsNameShort[len(defaultNs):].strip('.')

        for name, func in inspect.getmembers(ns):
            if not callable(func):
                continue
            if name.startswith('_'):
                continue

            fullName = nsName + '.' + name if nsName else name
            shortName = (nsNameShort + '.' + name) if len(nsNameShort) > 0 else name
            
            if fullName in functionMap:
                print(f"Warning: Function '{fullName}' already exists in jnpFunctions. Overwriting.")
            functionMap[fullName] = func

            
            if fullName != shortName and shortName not in functionMap:
                functionMap[shortName] = func

        for name, object in inspect.getmembers(ns):
            if inspect.isclass(object) or inspect.ismodule(object):
                if name.startswith('_'):
                    continue
                recursiveSearchForFunctions(object)

    for ns in namespaces:
        recursiveSearchForFunctions(ns)

    return functionMap

def requiresRandomKey(func):
    try:
        signature = inspect.signature(func)
        parameters = list(signature.parameters.values())

        if parameters and parameters[0].name == 'key':
            return True
        
    except ValueError:
        pass
    
    return False
