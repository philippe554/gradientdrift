
import jax
import optax
import numpy as np
import scipy
import datetime
import re
import jax.numpy as jnp

class Model:
    def fitClosedForm(self, dataset, batchSize = 100):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def fit(self, dataset, seed = 42, batchSize = -1, maxNumberOfSteps = 1000, parameterUpdateFrequency = 1, optimizer = "ADAM"):

        if optimizer.lower() == "closedform":
            self.fitClosedForm(dataset, batchSize)
            return

        fitStartTime = datetime.datetime.now()

        self.fitConfig = {
            "Seed": seed,
            "Batch size": "Full batch" if batchSize == -1 else batchSize,
            "Max number of steps": maxNumberOfSteps,
            "Parameter update frequency": "Once per step" if parameterUpdateFrequency == -1 else "Every " + str(parameterUpdateFrequency) + " batches",
            "Fit start time": fitStartTime.strftime("%a, %d %b %Y %H:%M:%S"),
        }

        # Prepare the optimization environment
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        self.setRandomParameters(subkey)      

        print("Initial parameters:")
        for name, param in self.parameterValues.items():
            print(f"{name}: {param}")  

        if optimizer.lower() == "adam":
            optimizerObj = optax.adam(0.1)
            optimizerUsesState = False
            self.fitConfig["Optimizer"] = "ADAM"
            self.fitConfig["Learning rate"] = 0.01

        elif optimizer.lower() == "lbfgs":
            optimizerObj = optax.lbfgs()
            optimizerUsesState = True
            self.fitConfig["Optimizer"] = "L-BFGS"

        else:
            raise ValueError(f"Unknown optimizer: {optimizer}. Supported optimizers are 'adam' and 'lbfgs'.")
        
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

        if not optimizerUsesState:
            @jax.jit
            def calcGrad(params, batch):
                loss, grads = jax.value_and_grad(self.loss)(params, batch)
                return loss, grads
            
            @jax.jit
            def applyUpdate(grads, params, optState):
                updates, newOptState = optimizerObj.update(grads, optState)
                newParams = optax.apply_updates(params, updates)
                return newParams, newOptState
            
        else:
            @jax.jit
            def calcGradWithState(params, batch, optimizerState):
                loss, grads = optax.value_and_grad_from_state(self.loss)(params, batch, state = optimizerState)
                return loss, grads
        
            @jax.jit
            def compiledLoss(params, data):
                return self.loss(params, data)
            
            def applyUpdateWithLineSearch(grads, params, optState, loss, selectedBatches):
                def lossOverBatches(params):
                    lossSum = 0
                    for i in selectedBatches:
                        batch = dataset.getBatch(i)
                        lossSum += compiledLoss(params, batch.data)
                    return lossSum / len(selectedBatches)
                
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

            key, subkey = jax.random.split(key)
            batchOrder = jax.random.permutation(subkey, numberOfBatches)
            
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
                    self.parameterValues, optimizerState = applyUpdateWithLineSearch(avgGrads, self.parameterValues, optimizerState, avgBatchLoss, selectedBatches)                        
                
                aggregatedLoss = 0.0
                aggregatedGrads = jax.tree_util.tree_map(jax.numpy.zeros_like, self.parameterValues)
                aggregatedCount = 0
                selectedBatches = []

            for i in range(numberOfBatches):
                selectedBatches.append(batchOrder[i])
                batch = dataset.getBatch(batchOrder[i])
                
                if not optimizerUsesState:
                    loss, grads = calcGrad(self.parameterValues, batch.data)
                else:
                    loss, grads = calcGradWithState(self.parameterValues, batch.data, optimizerState)

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
            if pamamsChangeNorm < 1e-5:
                print("Convergence reached based on parameter change norm.")
                break

        fitEndTime = datetime.datetime.now()
        self.fitConfig["Fit end time"] = fitEndTime.strftime("%a, %d %b %Y %H:%M:%S")
        self.fitConfig["Fit duration"] = str(fitEndTime - fitStartTime)

    def loss(self, params, data):
        return -self.logLikelihood(params, data)
    
    def hessian(self, params, dataset):
        # Use JAX's built-in utility to flatten the pytree and get an un-flattening function.
        # This is much safer than doing it manually.
        flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params)

        hessian_for_batch = jax.jit(jax.hessian(lambda p, d: self.loss(unflatten_fn(p), d)))
        
        # Use jax.vmap to create a function that computes the gradient for each
        # observation in a batch and stacks them into a matrix.
        # in_axes=(None, 0) means we don't map over params, but we do map over data.    
        jacobian_for_batch  = jax.jit(
            jax.jacfwd(lambda p, d: self.loss(unflatten_fn(p), d, returnLossPerSample = True))
        )

        # --- Aggregation loop ---
        # (This part remains the same)
        aggregatedHessian = jax.numpy.zeros((flat_params.size, flat_params.size))
        opg_sum = jax.numpy.zeros((flat_params.size, flat_params.size))
        numberOfBatches = dataset.getNumberOfBatches()
        effectiveEntriesPerBatch = 4999
        for i in range(numberOfBatches):
            batch_data = dataset.getBatch(i).data
            batchHessian = hessian_for_batch(flat_params, batch_data)
            aggregatedHessian += batchHessian

            # 2. Aggregate the Outer-Product-of-Gradients
            # Get the gradient for EACH observation in the batch (returns a matrix)
            per_obs_gradients = jacobian_for_batch (flat_params, batch_data)
            if jnp.shape(per_obs_gradients) != (effectiveEntriesPerBatch, flat_params.size):
                raise ValueError(f"Expected per_obs_gradients to have shape ({effectiveEntriesPerBatch}, {flat_params.size}), but got {jnp.shape(per_obs_gradients)}. This might indicate a mismatch in the number of observations or parameters.")
            # The sum of outer products for the batch is g.T @ g
            opg_sum += per_obs_gradients.T @ per_obs_gradients

        avg_hessian = aggregatedHessian / numberOfBatches

        # Return both the Hessian and the function needed to unflatten the results
        return avg_hessian, opg_sum, unflatten_fn      
    
    def setRandomParameters(self, key):
        numberOfParameters = len(self.parameterValues)
        keys = jax.random.split(key, numberOfParameters)
        self.parameterValues = { name : jax.random.normal(key, jnp.shape(param)) * 0.1 for key, (name, param) in zip(keys, self.parameterValues.items()) }

    def getVariablePrettyName(self, variableName, indexList):
        if not hasattr(self, "paramDims") or variableName not in self.paramDims:
            return variableName + "[" + ".".join(map(str, indexList)) + "]"
        else:
            template = self.paramDims.get(variableName).copy()

            for i in range(len(indexList)):
                def replacer(match):
                    offsetStr = match.group(1)
                    
                    offset = 0
                    if offsetStr:
                        offset = int(offsetStr)

                    try:
                        baseValue = int(indexList[i])
                        return str(baseValue + offset)
                    except (ValueError, TypeError):
                        if offset != 0:
                            return match.group(0)
                        else:
                            return str(indexList[i])

                pattern = r'\[i([+-]\d+)?\]'
                
                if i < len(template):
                    template[i] = re.sub(pattern, replacer, template[i])


            return variableName + "[" + ".".join(template) + "]"

    def summary(self, dataset, trueParams = None):
        tableWidth = 104

        print("=" * tableWidth)
        print("Model Summary".center(tableWidth))
        
        print("=" * tableWidth)
        print(f"{'Model Type:':<20.20}{self.getModelType():<32.32}") # Assuming getModelType might be long
        print(f"{'Fit date:':<20.20}{str(self.getFitDate()):<32.32}{' LL:':<20.20}{str(self.getLogLikelihood(dataset)):<32.32}")
        print(f"{'Fit time:':<20.20}{str(self.getFitTime()):<32.32}{' AIC:':<20.20}{str(self.getAIC(dataset)):<32.32}")
        print(f"{'No. obs.:':<20.20}{str(self.getObsCount(dataset)):<32.32}{' BIC:':<20.20}{str(self.getBIC(dataset)):<32.32}")
        print(f"{'Std. err. method:':<20.20}{'Robust':<32.32}")

        print("=" * tableWidth)
        print("Fit configuration")
        print("-" * tableWidth)
        for key, value in self.fitConfig.items():
            key = key + ":"
            print(f"{key:<40.40} {str(value):<63.63}")
        
        print("=" * tableWidth)
        print("Parameter estimates")
        print("-" * tableWidth)
        print(f"{'Parameter':<28.28} {'Estimate':>10.10} {'Std. err.':>10.10} {'Z-stat.':>10.10} {'P-value':>9.9} {'Conf. interval':>21.21} {'True val':>10.10}")

        stdErrs = self.getStdErrs(dataset)
        zStats = self.getZStats(dataset)
        pValues = self.getPValues(dataset)
        lowerBounds, upperBounds = self.getConfIntervals(dataset)

        for paramName in self.parameterValues:
            paramArray = self.parameterValues[paramName]

            for index, estimate in np.ndenumerate(paramArray):

                if hasattr(self, "parameterizations") and paramName in self.parameterizations:
                    estimate = jax.nn.softplus(estimate)

                paramPrettyName = self.getVariablePrettyName(paramName, index)
                confInterval = f"{lowerBounds[paramName][index]:10.3f};{upperBounds[paramName][index]:10.3f}"
                pValueSymbol = "***" if pValues[paramName][index] < 0.001 else "** " if pValues[paramName][index] < 0.01 else "*  " if pValues[paramName][index] < 0.05 else "   "
                
                if trueParams and paramName in trueParams:
                    try:
                        if np.shape(trueParams[paramName]) == ():
                            trueValue = f"{trueParams[paramName]:10.3f}"
                        else:
                            trueValue = f"{trueParams[paramName][index]:10.3f}"
                    except (IndexError, TypeError):
                        trueValue = "N/A"
                else:
                    trueValue = "N/A"

                print(f"{paramPrettyName:<28.28} {estimate:>10.3f} {stdErrs[paramName][index]:>10.5f} {zStats[paramName][index]:>10.3f} {pValues[paramName][index]:>6.3f}{pValueSymbol} {confInterval:>21.21} {trueValue:>10.10}")
        print(" " * 31 + "P-value symbols: *** for p < 0.001, ** for p < 0.01, * for p < 0.05")
        print("=" * tableWidth)
        

        
    
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

    def getCoefs(self):
        return self.parameterValues
    
    def getTrueParams(self):
        return {
            'mu': self.parameterValues['mu'],
            'omega': jax.nn.softplus(self.parameterValues['omega']),
            'alpha': jax.nn.softplus(self.parameterValues['alpha']),
            'beta': jax.nn.softplus(self.parameterValues['beta'])
        }
    
    def getStdErrs(self, dataset):
        """
        Calculates the standard errors of the model parameters using the robust
        ravel_pytree utility to prevent ordering bugs.
        """
        # Get both the Hessian and the unflattening function
        hessianOfMean, opg_matrix, unflatten_fn = self.hessian(self.parameterValues, dataset)
        
        if hessianOfMean is None: # Handle potential failure in hessian calculation
            return jax.tree_util.tree_map(lambda p: jax.numpy.full_like(p, jax.numpy.nan), self.parameterValues)

        # Scale to get the Hessian of the sum
        hessianOfSum = hessianOfMean * dataset.getEffectiveNObs()
        
        try:
            invHessian = jax.numpy.linalg.inv(hessianOfSum)

            # covMatrix = invHessian
            robustCovMatrix = invHessian @ opg_matrix @ invHessian

            # APPLY FINITE SAMPLE CORRECTION
            correction_factor = dataset.getEffectiveNObs() / (dataset.getEffectiveNObs() - len(self.parameterValues))
            robustCovMatrix *= correction_factor

            ### Correct for parameterization

            paramsFlat, unravelFunc = jax.flatten_util.ravel_pytree(self.parameterValues)

            def getTrueParamsFlat(paramsFlat):
                params = unravelFunc(paramsFlat)

                p = {
                    'mu': params['mu'],
                    'omega': jax.nn.softplus(params['omega']),
                    'alpha': jax.nn.softplus(params['alpha']),
                    'beta': jax.nn.softplus(params['beta'])
                }

                return jax.flatten_util.ravel_pytree(p)[0]
            
            jacobian = jax.jacfwd(getTrueParamsFlat)(paramsFlat)

            robustCovMatrix = jacobian @ robustCovMatrix @ jacobian.T

            ###

            stdErrorsFlat = jax.numpy.sqrt(jax.numpy.diag(robustCovMatrix))
            
            # Use the unflattening function to safely convert the flat vector
            # back into the correct PyTree structure. No manual reshaping!
            stdErrorsPytree = unflatten_fn(stdErrorsFlat)
            
            return stdErrorsPytree
            
        except jax.numpy.linalg.LinAlgError:
            print("Warning: Hessian is not invertible. Standard errors cannot be computed.")
            return jax.tree_util.tree_map(lambda p: jax.numpy.full_like(p, jax.numpy.nan), self.parameterValues)
        
    def getZStats(self, dataset):
        """
        Calculates the z-statistics for each parameter (estimate / std_err).
        """
        estimates = self.getTrueParams()
        std_errs = self.getStdErrs(dataset)
        
        # jax.tree_map can operate on multiple pytrees with the same structure
        # It will pass the corresponding leaf from each tree to the lambda function.
        z_stats = jax.tree_util.tree_map(
            lambda estimate, se: estimate / se, 
            estimates, 
            std_errs
        )
        return z_stats

    def getPValues(self, dataset):
        """
        Calculates the two-tailed p-values for the z-statistics.
        """
        z_stats = self.getZStats(dataset)
        
        # The p-value for a two-tailed test is 2 * (1 - CDF(|z|)).
        # JAX's survival function `norm.sf` is 1 - CDF, so this is 2 * norm.sf(|z|).
        # p_values = jax.tree_util.tree_map(
        #     lambda z: 2 * jax.scipy.stats.norm.sf(jax.numpy.abs(z)),
        #     z_stats
        # )

        def t_sf(t, df):
            x = df / (df + t**2)
            a = df / 2.0
            b = 0.5
            return 0.5 * jax.scipy.special.betainc(a, b, x)

        t_stats = z_stats
        degrees_of_freedom = dataset.getEffectiveNObs() - len(self.parameterValues)

        p_values = jax.tree_util.tree_map(
            lambda t_stat: 2 * t_sf(jnp.abs(t_stat), df=degrees_of_freedom),
            t_stats
        )


        return p_values

    def getConfIntervals(self, dataset):
        estimates = self.getTrueParams()
        std_errs = self.getStdErrs(dataset)        
        
        # critical_value = 1.96 # z critical value for 95% CI, assuming normal distribution
        
        degrees_of_freedom = dataset.getEffectiveNObs() - len(self.parameterValues)
        critical_value = scipy.stats.t.ppf(0.975, df=degrees_of_freedom) # t critical value for 95% CI, using degrees of freedom from the dataset
        
        lowerBound = jax.tree_util.tree_map(lambda estimate, se: estimate - critical_value * se, estimates, std_errs)
        upperBound = jax.tree_util.tree_map(lambda estimate, se: estimate + critical_value * se, estimates, std_errs)
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
        
    def summary_old(self, batch, trueParams=None):
        print("\n--- Model Summary ---")
        stdErrors = self.stdError(self.parameterValues, batch)
        if stdErrors is None:
            print("Summary could not be generated as standard errors could not be computed.")
            return

        # Prepare table header
        header_items = ["Parameter", "Estimate", "Std. Error"]
        header_format = "{:<20} {:>12} {:>12}"
        if trueParams:
            header_items.insert(1, "True Value")
            header_format += " {:>12}"
        
        # Print Header
        print(header_format.format(*header_items))
        print("-" * (len(header_format.format(*header_items)) + 2))

        # Iterate through all parameters in the pytree (e.g., 'coeffs', 'const')
        for key in self.parameterValues:
            param_array = self.parameterValues[key]
            se_array = stdErrors[key]
            true_array = trueParams.get(key) if trueParams is not None else None

            # Use ndenumerate to handle any parameter shape (scalar, vector, matrix, etc.)
            for index, estimate in np.ndenumerate(param_array):
                # Create a readable parameter name like "coeffs[0,0,0]"
                param_name = f"{key}{list(index)}"
                se = se_array[index]
                
                row_items = [param_name, f"{estimate:.4f}", f"{se:.4f}"]
                
                if true_array is not None:
                    # Check if the true parameter for this key exists before accessing
                    try:
                        true_val = true_array[index]
                        row_items.insert(1, f"{true_val:.4f}")
                    except (TypeError, IndexError):
                         # Handle cases where true_val is not subscriptable (e.g. for logSigma)
                        row_items.insert(1, "N/A")

                print(header_format.format(*row_items))

        # TODO: calc AIC and BIC, question: should the sigma be included in the AIC/BIC calculation?