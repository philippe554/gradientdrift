
import jax
import optax
import numpy as np
import scipy
import datetime
import re
import jax.numpy as jnp
from functools import partial
import numbers
from gradientdrift.math.ess import *
from gradientdrift.math.rhat import *

class Model:
    def predict(self, dataset = None, steps = None, key = jax.random.PRNGKey(42)):
        key1, key2 = jax.random.split(key)
        if dataset is not None:
            self.constructModel(dataset.getDataShape(), key1)
            states = self.predictStep(self.parameterValues, randomKey = key2, data = dataset.data)
        else:
            self.constructModel({}, key1)
            states = self.predictStep(self.parameterValues, randomKey = key2, steps = steps)
        states = {k.replace("model.", ""): v for k, v in states.items()}
        return states

    def loss(self, dataset, returnLossPerSample = False):
        if dataset is None:
            raise ValueError("Dataset must be provided to compute loss.")
        self.constructModel(dataset.getDataShape())
        return self.lossStep(self.parameterValues, dataset.data, returnLossPerSample)
    
    def MCMC_getCredibleIntervals(self, allParams, confidenceLevel = 0.95):
        lowerQuantile = (1.0 - confidenceLevel) / 2.0  # e.g., 0.025
        upperQuantile = (1.0 + confidenceLevel) / 2.0  # e.g., 0.975
        quantilesToFind = jnp.array([lowerQuantile, upperQuantile])

        def getLeafInterval(arr):
            nChains = arr.shape[0]
            nSamples = arr.shape[1]

            mergedParams = arr.reshape((nChains * nSamples, -1))

            intervals = jnp.quantile(mergedParams, q=quantilesToFind, axis=0)

            originalParamShape = arr.shape[2:]

            return intervals.reshape((2, *originalParamShape))
        
        allIntervals = jax.tree_util.tree_map(getLeafInterval, allParams)

        lowerBounds = jax.tree_util.tree_map(lambda x: x[0], allIntervals)
        upperBounds = jax.tree_util.tree_map(lambda x: x[1], allIntervals)

        return lowerBounds, upperBounds

    def MCMC_fit(self, dataset, seed):
        self.requestPadding(dataset)
        dataset.prepareBatches(-1)
        data = dataset.getBatch(0).data

        burnInChunkSize = 100
        burnInChunks = 100
        targetAcceptanceRate = 0.234 # Optimal for high-dimensional models
        stepSizeAdaptationFactor = 1.2

        chunkSize = 10000
        numberOfChunks = 100
        convergenceCheckInterval = 5   
        numberOfChains = 8

        key = jax.random.PRNGKey(seed)
        initKey, key = jax.random.split(key)

        initKeys = jax.random.split(initKey, numberOfChains)
        lastParams = jax.vmap(lambda k: self.getNewParamsFromPrior(k))(initKeys)

        dummyParams = self.getNewParamsFromPrior(jax.random.PRNGKey(0))
        stepSizePerBlock = []
        for name in dummyParams.keys():
            stepSize = jax.tree_util.tree_map(jax.numpy.zeros_like, dummyParams)
            stepSize[name] = jnp.ones_like(dummyParams[name]) * 0.1
            stepSizePerBlock.append(stepSize)

        vmapped_MCMC_chunk = jax.vmap(self.MCMC_chunk, in_axes=(0, None, None, 0, None))

        # Burn-in loop

        for step in range(burnInChunks):
            chunkKey, key = jax.random.split(key)
            subkeys = jax.random.split(chunkKey, numberOfChains)

            stepSizePerBlockPytree = jax.tree_util.tree_map(
                lambda *leaves: jnp.stack(leaves), 
                *stepSizePerBlock
            )
            lastParams, chunkParams, acceptCounts = vmapped_MCMC_chunk(lastParams, data, stepSizePerBlockPytree, subkeys, burnInChunkSize)

            acceptCounts = jnp.mean(acceptCounts, axis=0)

            for i, acceptRate in enumerate(acceptCounts / burnInChunkSize):
                if acceptRate < targetAcceptanceRate:
                    stepSizePerBlock[i] = jax.tree_util.tree_map(lambda s: s / stepSizeAdaptationFactor, stepSizePerBlock[i])
                else:
                    stepSizePerBlock[i] = jax.tree_util.tree_map(lambda s: s * stepSizeAdaptationFactor, stepSizePerBlock[i])

        print(f"Completed burn-in, Acceptance rate: {acceptCounts / burnInChunkSize}")

        # Main MCMC loop

        allParamsPerChain = []

        for step in range(numberOfChunks):
            chunkKey, key = jax.random.split(key)
            subkeys = jax.random.split(chunkKey, numberOfChains)

            stepSizePerBlockPytree = jax.tree_util.tree_map(
                lambda *leaves: jnp.stack(leaves), 
                *stepSizePerBlock
            )
            lastParams, chunkParams, acceptCounts = vmapped_MCMC_chunk(lastParams, data, stepSizePerBlockPytree, subkeys, chunkSize)

            acceptCounts = jnp.mean(acceptCounts, axis=0)

            allParamsPerChain.append(chunkParams)
            print(f"Completed MCMC chunk {step + 1}, Acceptance rate: {acceptCounts / chunkSize}")

            if (step + 1) % convergenceCheckInterval == 0:
                allParamsMerged = jax.tree_util.tree_map(
                    lambda *chunks: jnp.concatenate(chunks, axis=1), 
                    *allParamsPerChain
                )
                
                rHatValues = getRHat(allParamsMerged)
                ess = calculate_ess(allParamsMerged)

                maxRHat = max(
                    jnp.max(rHatArray) 
                    for rHatArray in jax.tree_util.tree_leaves(rHatValues)
                )

                minESS = min(
                    jnp.min(essArray) 
                    for essArray in jax.tree_util.tree_leaves(ess)
                )

                print(f"MCMC convergence check at chunk {step + 1}: max R-hat = {maxRHat:.4f}, min ESS = {minESS:.4f}")
                if maxRHat < 1.01 and minESS > 200:
                    print("MCMC chains have converged based on R-hat criterion.")
                    break

        allParamsPerChain = jax.tree_util.tree_map(
            lambda *chunks: jnp.concatenate(chunks, axis=1), 
            *allParamsPerChain
        )

        self.parameterValues = jax.tree_util.tree_map(
            lambda arr: jnp.mean(arr, axis=(0, 1)), 
            allParamsPerChain
        )

        self.MCMC_credibleIntervals = self.MCMC_getCredibleIntervals(allParamsPerChain)

        if numberOfChains > 1:
            rHatValues = getRHat(allParamsPerChain)

            print("MCMC R-hat values:")
            for paramName, rHatArray in rHatValues.items():
                for index, rHat in np.ndenumerate(np.asarray(rHatArray)):
                    paramPrettyName = self.getVariablePrettyName(paramName, index)
                    print(f"{paramPrettyName:<30.30} R-hat: {rHat:.4f}")

    def fit(self, dataset, seed = 42, batchSize = -1, maxNumberOfSteps = 1000, parameterUpdateFrequency = -1, optimizer = "ADAM", burnInTime = 0, learningRate = None):
        self.burnInTime = burnInTime
        
        try:
            key = jax.random.PRNGKey(seed)

            self.constructModel(dataset.getDataShape(), key)
            
            fitStartTime = datetime.datetime.now()

            self.fitConfig = {
                "Seed": seed,
                "Batch size": "Full batch" if batchSize == -1 else batchSize,
                "Max number of steps": maxNumberOfSteps,
                "Parameter update frequency": "Once per step" if parameterUpdateFrequency == -1 else "Every " + str(parameterUpdateFrequency) + " batches",
                "Fit start time": fitStartTime.strftime("%a, %d %b %Y %H:%M:%S"),
            }   

            if optimizer.lower() == "adam":
                learningRate = learningRate if learningRate is not None else 0.01
                optimizerObj = optax.adam(learningRate)
                optimizerUsesState = False
                self.fitConfig["Optimizer"] = "ADAM"
                self.fitConfig["Learning rate"] = learningRate

            elif optimizer.lower() == "lbfgs":
                optimizerObj = optax.lbfgs()
                optimizerUsesState = True
                self.fitConfig["Optimizer"] = "L-BFGS"

            elif optimizer.lower() == "mcmc":
                self.MCMC_fit(dataset, seed)
                fitEndTime = datetime.datetime.now()
                self.fitConfig["Fit end time"] = fitEndTime.strftime("%a, %d %b %Y %H:%M:%S")
                self.fitConfig["Fit duration"] = str(fitEndTime - fitStartTime)
                return

            else:
                raise ValueError(f"Unknown optimizer: {optimizer}.")
            
            optimizerState = optimizerObj.init(self.parameterValues)
            
            # Prepare the dataset
            self.requestPadding(dataset)
            dataset.prepareBatches(batchSize)
            numberOfSamples = dataset.getEffectiveNObs()
            numberOfBatches = dataset.getNumberOfBatches()
            if numberOfBatches == 0:
                raise ValueError("No batches available. Check if the dataset is properly prepared and has enough data.")

            if parameterUpdateFrequency == -1:
                parameterUpdateFrequency = numberOfBatches

            ### =============

            if len(self.OLS) > 0 and len(self.latentParameters) == 0:
                print("Running OLS optimization...")

                XXSum = {}
                XYSum = {}
                for i in range(numberOfBatches):
                    batch = dataset.getBatch(i)
                    X, Y = self.OLS_getXY(batch.data)
                    for dependingVariable in self.OLS:
                        if dependingVariable not in XXSum:
                            XXSum[dependingVariable] = jnp.transpose(X[dependingVariable]) @ X[dependingVariable]
                            XYSum[dependingVariable] = jnp.transpose(X[dependingVariable]) @ Y[dependingVariable]
                        else:
                            XXSum[dependingVariable] += jnp.transpose(X[dependingVariable]) @ X[dependingVariable]
                            XYSum[dependingVariable] += jnp.transpose(X[dependingVariable]) @ Y[dependingVariable]

                B = {}
                for dependingVariable in self.OLS:
                    B[dependingVariable] = jax.numpy.linalg.solve(XXSum[dependingVariable], XYSum[dependingVariable])
                    
                    index = 0
                    for term in self.OLS[dependingVariable]["terms"]:
                        shape = self.parameterValues[term].shape
                        size = np.prod(shape)
                        self.parameterValues[term] = jnp.reshape(B[dependingVariable][index:index + size], self.parameterValues[term].shape)
                        index += size

                    if self.OLS[dependingVariable]["bias"] is not None:
                        bias = jnp.reshape(B[dependingVariable][-1], (1,))
                        self.parameterValues[self.OLS[dependingVariable]["bias"]] = bias

                RSS = {}
                for i in range(numberOfBatches):
                    batch = dataset.getBatch(i)
                    X, Y = self.OLS_getXY(batch.data)
                    for dependingVariable in self.OLS:
                        error = Y[dependingVariable] - X[dependingVariable] @ B[dependingVariable]
                        if dependingVariable not in RSS:
                            RSS[dependingVariable] = jnp.sum(error**2)
                        else:
                            RSS[dependingVariable] += jnp.sum(error**2)
                
                df = self.getDegreesOfFreedom(dataset)
                for dependingVariable in self.OLS:
                    sigma = jnp.sqrt(RSS[dependingVariable] / df)
                    sigmaName = "model.sigma_" + dependingVariable
                    sigmaParameterized = jnp.reshape(self.parameterizations[sigmaName]["inverse"](sigma), self.parameterValues[sigmaName].shape)
                    self.parameterValues[sigmaName] = sigmaParameterized

            else:
                print("Running gradient descent optimization...")
                
                if not optimizerUsesState:
                    @jax.jit
                    def calcGrad(params, batch):
                        loss, grads = jax.value_and_grad(self.lossStep)(params, batch)
                        return loss, grads
                    
                    @jax.jit
                    def applyUpdate(grads, params, optState):
                        updates, newOptState = optimizerObj.update(grads, optState)
                        newParams = optax.apply_updates(params, updates)
                        return newParams, newOptState
                    
                else:
                    @jax.jit
                    def calcGradWithState(params, batch, optimizerState):
                        loss, grads = optax.value_and_grad_from_state(self.lossStep)(params, batch, state = optimizerState)
                        return loss, grads
                    
                    @jax.jit
                    def applyUpdateWithLineSearch(grads, params, optState, loss, batches):
                        def lossOverBatches(params):
                            batchLosses = jax.vmap(self.lossStep, in_axes=(None, 0))(params, batches)
                            return jax.numpy.mean(batchLosses)
                        
                        updates, newOptState = optimizerObj.update(
                            grads, optState, params, value = loss, grad = grads, value_fn = lossOverBatches
                        )
                        
                        newParams = optax.apply_updates(params, updates)
                        return newParams, newOptState
                
                previousStepLoss = float('inf')
                
                # Run the optimization loop 
                for step in range(maxNumberOfSteps):

                    self.fitConfig["Number of steps"] = step + 1

                    stepStartParams = jax.tree_util.tree_map(lambda x: x, self.parameterValues)

                    stepLoss = 0.0
                    stepGrads = jax.tree_util.tree_map(jax.numpy.zeros_like, self.parameterValues)

                    key, subKey = jax.random.split(key)
                    batchOrder = jax.random.permutation(subKey, numberOfBatches)
                    
                    aggregatedLoss = 0.0
                    aggregatedGrads = jax.tree_util.tree_map(jax.numpy.zeros_like, self.parameterValues)
                    aggregatedCount = 0
                    selectedBatches = []

                    def applyHelper():
                        nonlocal aggregatedGrads, aggregatedCount, selectedBatches, aggregatedLoss, optimizerState
                        if aggregatedCount == 0:
                            raise ValueError("Aggregated count should not be zero at this point.")
                        if aggregatedCount > 1:
                            avgGrads = jax.tree_util.tree_map(lambda g: g / aggregatedCount, aggregatedGrads)
                        else:
                            avgGrads = aggregatedGrads

                        avgBatchLoss = aggregatedLoss / aggregatedCount

                        if not optimizerUsesState:
                            self.parameterValues, optimizerState = applyUpdate(avgGrads, self.parameterValues, optimizerState)
                        else:
                            batches = [dataset.getBatch(i).data for i in selectedBatches]
                            stackedBatches = jax.tree_util.tree_map(lambda *args: jnp.stack(args, axis=0), *batches)
                            self.parameterValues, optimizerState = applyUpdateWithLineSearch(
                                avgGrads, self.parameterValues, optimizerState, avgBatchLoss, stackedBatches)                        
                            
                        aggregatedLoss = 0.0
                        aggregatedGrads = jax.tree_util.tree_map(jax.numpy.zeros_like, self.parameterValues)
                        aggregatedCount = 0
                        selectedBatches = []

                    for i in range(numberOfBatches):
                        selectedBatches.append(batchOrder[i])
                        batch = dataset.getBatch(batchOrder[i])

                        # jaxpr = jax.make_jaxpr(self.lossStep)(self.parameterValues, batch.data)
                        # print(jaxpr)
                        # exit()

                        if not optimizerUsesState:
                            loss, grads = calcGrad(self.parameterValues, batch.data)
                        else:
                            loss, grads = calcGradWithState(self.parameterValues, batch.data, optimizerState)

                        lossContainsNaN = jnp.any(jnp.isnan(loss))
                        gradContainsNaN = any(jnp.any(jnp.isnan(g)) for g in jax.tree_util.tree_leaves(grads))
                        if lossContainsNaN or gradContainsNaN:
                            print(f"\n\nError: Loss or gradients contain NaN at step {step + 1}, batch {i + 1}.")
                            print(f"Gradients: {grads}")
                            exit(1)

                        stepLoss += loss * batch.getEffectiveNObs()
                        stepGrads = jax.tree_util.tree_map(jax.numpy.add, stepGrads, grads)

                        aggregatedLoss += loss
                        aggregatedGrads = jax.tree_util.tree_map(jax.numpy.add, aggregatedGrads, grads)
                        aggregatedCount += 1
                        
                        if aggregatedCount == parameterUpdateFrequency:
                            applyHelper()

                    if aggregatedCount > 0:
                        applyHelper()

                    sampleLoss = stepLoss / numberOfSamples  
                    print(f"Step {step+1:4d}, Loss: {sampleLoss:.6f}")

                    lossImprovement = abs(previousStepLoss - sampleLoss)
                    previousStepLoss = sampleLoss
                    if lossImprovement < 1e-9:
                        print("Convergence reached based on loss threshold.")
                        break

                    stepGrads = jax.tree_util.tree_map(lambda g: g / numberOfBatches, stepGrads)
                    grad_norm = jax.numpy.linalg.norm(jax.tree_util.tree_flatten(stepGrads)[0][0])

                    if grad_norm < 1e-7:
                        print("Convergence reached based on gradient norm.")
                        break

                    pamamsChange = jax.tree_util.tree_map(lambda new, old: new - old, self.parameterValues, stepStartParams)
                    pamamsChangeNorm = jax.numpy.linalg.norm(jax.tree_util.tree_flatten(pamamsChange)[0][0])
                    if pamamsChangeNorm < 1e-7:
                        print("Convergence reached based on parameter change norm.")
                        break

        except KeyboardInterrupt:
            print("Optimization interrupted by user. Press Ctrl+C again to exit.")

        fitEndTime = datetime.datetime.now()
        self.fitConfig["Fit end time"] = fitEndTime.strftime("%a, %d %b %Y %H:%M:%S")
        self.fitConfig["Fit duration"] = str(fitEndTime - fitStartTime)

    def constructModel(self):
        _, self.unflattenFunc = jax.flatten_util.ravel_pytree(self.parameterValues)
    
    def flattenParameters(self, parameters):
        parametersFlat, _ = jax.flatten_util.ravel_pytree(parameters)
        return parametersFlat
    
    def unflattenParameters(self, parametersFlat):
        return self.unflattenFunc(parametersFlat)
    
    def calculateHessianAndOPGMatrix(self, params, dataset):
        hessianFunc = jax.jit(
            jax.hessian(
                lambda p, d: self.lossStep(self.unflattenParameters(p), d)
            )
        )
        jacobianFunc = jax.jit(
            jax.jacfwd(
                lambda p, d: self.lossStep(self.unflattenParameters(p), d, returnLossPerSample = True)
            )
        )

        parametersFlat = self.flattenParameters(params)
        hessianSum = jax.numpy.zeros((parametersFlat.size, parametersFlat.size))
        OPGSum = jax.numpy.zeros((parametersFlat.size, parametersFlat.size))
        numberOfBatches = dataset.getNumberOfBatches()
        
        for i in range(numberOfBatches):
            data = dataset.getBatch(i).data
            hessianSum += hessianFunc(parametersFlat, data)
            OPGBatch = jacobianFunc(parametersFlat, data)
            OPGSum += OPGBatch.T @ OPGBatch

        hessian = hessianSum / numberOfBatches
        return hessian, OPGSum      
    
    def getVariablePrettyName(self, variableName, indexList):
        if variableName not in self.parameters:
            raise ValueError(f"Variable '{variableName}' not found in model parameters.")
        
        variableNameShort = variableName
        if variableName.startswith("model."):
            variableNameShort = variableName[6:]
        
        indexList = list(indexList)
        if "dimensionLabels" in self.parameters[variableName]:
            for i, values in self.parameters[variableName]["dimensionLabels"].items():
                indexList[i] = values[indexList[i]]
        
        if "shape" in self.parameters[variableName]:
            shape = self.parameters[variableName]["shape"]
            if shape == (1,):
                return variableNameShort
 
        return variableNameShort + "[" + ".".join(map(str, indexList)) + "]"
        
    def summary(self, dataset, trueParams = None, confidenceLevel = 0.95):
        self.allInConfidenceInterval = True

        standardErrors = self.getStdErrs(dataset)
        
        tableWidth = 104

        print("=" * tableWidth)
        print("Model Summary".center(tableWidth))
        
        print("=" * tableWidth)
        print(f"{'Model Type:':<20.20}{self.getModelType():<32.32}") # Assuming getModelType might be long
        print(f"{'Fit date:':<20.20}{str(self.getFitDate()):<32.32}{' LL:':<20.20}{str(self.getLogLikelihood(dataset)):<32.32}")
        print(f"{'Fit time:':<20.20}{str(self.getFitTime()):<32.32}{' AIC:':<20.20}{str(self.getAIC(dataset)):<32.32}")
        print(f"{'No. obs.:':<20.20}{str(self.getObsCount(dataset)):<32.32}{' BIC:':<20.20}{str(self.getBIC(dataset)):<32.32}")
        print(f"{'Covariance est.:':<20.20}{'Robust':<32.32}")

        print("=" * tableWidth)
        print("Fit configuration")
        print("-" * tableWidth)
        for key, value in self.fitConfig.items():
            key = key + ":"
            print(f"{key:<40.40} {str(value):<63.63}")
        
        print("=" * tableWidth)
        print("Parameter estimates")
        print("-" * tableWidth)
        confidenceIntervalHeader = f"Conf. int. ({confidenceLevel * 100:.1f}%)"
        print(f"{'Parameter':<28.28} {'Estimate':>10.10} {'Std. err.':>10.10} {'t-stat.':>10.10} {'P-value':>9.9} {confidenceIntervalHeader:>21.21} {'True val':>10.10}")

        estimates = self.getConstraintParameters()
        tStats = self.getTStats(dataset, standardErrors = standardErrors)
        pValues = self.getPValues(dataset, standardErrors = standardErrors)
        
        if hasattr(self, "MCMC_credibleIntervals"):
            lowerBounds, upperBounds = self.MCMC_credibleIntervals
            lowerBounds = self.applyParameterization(lowerBounds)
            upperBounds = self.applyParameterization(upperBounds)
        else:
            lowerBounds, upperBounds = self.getConfIntervals(dataset, standardErrors = standardErrors, confidenceLevel = confidenceLevel)

        for paramName, paramArray in estimates.items():
            for index, estimate in np.ndenumerate(paramArray):
                paramPrettyName = self.getVariablePrettyName(paramName, index)
                confInterval = f"{lowerBounds[paramName][index]:10.3f};{upperBounds[paramName][index]:10.3f}"
                pValueSymbol = "***" if pValues[paramName][index] < 0.001 else "** " if pValues[paramName][index] < 0.01 else "*  " if pValues[paramName][index] < 0.05 else "   "
                
                if trueParams:
                    if paramName in trueParams:
                        trueValue = trueParams[paramName]
                    elif "." in paramName and paramName.split(".")[1] in trueParams:
                        trueValue = trueParams[paramName.split(".")[1]]
                    else:
                        trueValue = None

                    if trueValue is not None:
                        if type(trueValue) == np.ndarray:
                            trueValueIndexed = trueValue[index]
                        elif type(trueValue) == list and len(index) == 1:
                            trueValueIndexed = trueValue[index[0]]
                        elif isinstance(trueValue, numbers.Number):
                            trueValueIndexed = trueValue
                        else:
                            raise ValueError(f"Unsupported type for true parameter value: {type(trueValue)}")
                        
                        inConfidenceInterval = lowerBounds[paramName][index] <= trueValueIndexed and trueValueIndexed <= upperBounds[paramName][index]
                        
                        if not inConfidenceInterval:
                            self.allInConfidenceInterval = False
                        try:
                            trueValue = f"{trueValueIndexed:9.3f}"
                            if inConfidenceInterval:
                                trueValue += "*"
                            else:
                                trueValue += " "
                        except (IndexError, TypeError):
                            trueValue = "ERR "
                    else:
                        trueValue = "N/A "
                else:
                    trueValue = "N/A "

                print(f"{paramPrettyName:<28.28} {estimate:>10.3f} {standardErrors[paramName][index]:>10.5f} {tStats[paramName][index]:>10.3f} {pValues[paramName][index]:>6.3f}{pValueSymbol} {confInterval:>21.21} {trueValue:>10.10}")
        print(" " * 31 + "P-value symbols: *** for p < 0.001, ** for p < 0.01, * for p < 0.05")
        print("=" * tableWidth)      
    
    def isAllInConfidenceInterval(self):
        if hasattr(self, "allInConfidenceInterval"):
            return self.allInConfidenceInterval
        else:
            raise ValueError("Model does not have confidence interval information. Run Summary() first.")

    def getModelType(self):
        return "<TBD>"

    def getFitDate(self):
        return "<TBD>"

    def getFitTime(self):
        return "<TBD>"

    def getObsCount(self, dataset):
        return dataset.getEffectiveNObs()

    def getLogLikelihood(self, dataset):
        return "<TBD>"

    def getAIC(self, dataset):
        return "<TBD>"

    def getBIC(self, dataset):
        return "<TBD>"
    
    def applyParameterization(self, params):
        if not hasattr(self, "parameterizations"):
            return params
        else:
            constraintParams = {}
            for name, value in params.items():
                if name in self.parameterizations:
                    unconstrainedParamNames = self.parameterizations[name]["unconstraintParameterNames"]
                    unconstrainedParams = [params[unconstrainedParamName] for unconstrainedParamName in unconstrainedParamNames]
                    constraintParams[name] = self.parameterizations[name]["apply"](*unconstrainedParams)
                else:
                    constraintParams[name] = value
            return constraintParams
    
    def getConstraintParameters(self):
        return self.applyParameterization(self.parameterValues)
        
    def getDegreesOfFreedom(self, dataset):
        nObs = dataset.getEffectiveNObs()
        nParams = self.flattenParameters(self.parameterValues).size
        return nObs - nParams
    
    def getStdErrs(self, dataset):
        print("Calculating standard errors...")

        if len(self.latentParameters) > 0:
            print("Warning: Standard errors are not available for models with latent parameters.")
            return jax.tree_util.tree_map(lambda x: jnp.full_like(x, jnp.nan), self.getConstraintParameters())

        try:
            # Calculate the Hessian and OPG matrix
            hessian, OPGMatrix = self.calculateHessianAndOPGMatrix(self.parameterValues, dataset)
            hessian *= dataset.getEffectiveNObs()
        
            # Calculate the inverse of the Hessian
            invHessian = jax.numpy.linalg.inv(hessian)

            # Calculate the robust covariance matrix
            robustCovMatrix = invHessian @ OPGMatrix @ invHessian

            # Apply the correction factor for the number of observations
            N = dataset.getEffectiveNObs()
            dof = self.getDegreesOfFreedom(dataset)
            correctionFactor = N / dof
            robustCovMatrix *= correctionFactor

            # Correct for parameterization
            def helper(parametersFlat):
                params = self.unflattenParameters(parametersFlat)
                p = self.applyParameterization(params)
                return self.flattenParameters(p)
            jacobianFunc = jax.jacfwd(helper)
            parametersFlat = self.flattenParameters(self.parameterValues)
            jacobian = jacobianFunc(parametersFlat)
            robustCovMatrix = jacobian.T @ robustCovMatrix @ jacobian

            # Flatten the covariance matrix to get standard errors
            stdErrorsFlat = jax.numpy.sqrt(jax.numpy.diag(robustCovMatrix))
            
            # Unflatten the standard errors to match the structure of the parameters
            return self.unflattenParameters(stdErrorsFlat)
            
        except jax.numpy.linalg.LinAlgError:
            print("Warning: Hessian is not invertible. Standard errors cannot be computed.")
            exit(1)
        
    def getTStats(self, dataset, standardErrors = None):
        estimates = self.getConstraintParameters()
        if standardErrors is None:
            standardErrors = self.getStdErrs(dataset)
        
        return jax.tree_util.tree_map(lambda estimate, se: estimate / se, estimates, standardErrors)

    def getPValues(self, dataset, standardErrors = None):
        tStat = self.getTStats(dataset, standardErrors = standardErrors)
        dof = self.getDegreesOfFreedom(dataset)

        # TODO: check this function
        def t_sf(t, df):
            x = df / (df + t**2)
            a = df / 2.0
            b = 0.5
            return 0.5 * jax.scipy.special.betainc(a, b, x)

        return jax.tree_util.tree_map(lambda tStat: 2 * t_sf(jnp.abs(tStat), df = dof), tStat)

    def getConfIntervals(self, dataset, standardErrors = None, confidenceLevel = 0.95):
        estimates = self.getConstraintParameters()
        if standardErrors is None:
            standardErrors = self.getStdErrs(dataset)
                
        dof = self.getDegreesOfFreedom(dataset)

        criticalValue = scipy.stats.t.ppf((1 + confidenceLevel) / 2, df = dof)

        lowerBound = jax.tree_util.tree_map(lambda estimate, se: estimate - criticalValue * se, estimates, standardErrors)
        upperBound = jax.tree_util.tree_map(lambda estimate, se: estimate + criticalValue * se, estimates, standardErrors)
        return lowerBound, upperBound

    def getLjungBoxStat(self, dataset):
        return "<TBD>"
    
    def getLjungBoxPVal(self, dataset):
        return "<TBD>"
        
    def getJarqueBeraStat(self, dataset):
        return "<TBD>"
        
    def getJarqueBeraPVal(self, dataset):
        return "<TBD>"
        
    def getRootsOfCharPoly(self):
        return "<TBD>"
        
    def isStationary(self):
        return "<TBD>"
