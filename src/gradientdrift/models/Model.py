
import jax
import optax
import numpy as np

class Model:
    def fit(self, dataset, seed = 42, batchSize = -1, numberOfSteps = 100, parameterUpdateFrequency = 1):

        # Prepare the optimization environment
        key = jax.random.PRNGKey(seed)
        self.setRandomParameters(key)
        optimizer = optax.adam(0.01)
        optimizerState = optimizer.init(self.params)
        
        # Prepare the dataset
        self.requestPadding(dataset)
        dataset.prepareBatches(batchSize)
        numberOfBatches = dataset.getNumberOfBatches()
        if numberOfBatches == 0:
            raise ValueError("No batches available. Check if the dataset is properly prepared and has enough data.")

        if parameterUpdateFrequency == 1:

            @jax.jit
            def updateStep(current_params, current_opt_state, data):
                loss, grads = jax.value_and_grad(self.loss)(current_params, data)
                updates, new_opt_state = optimizer.update(grads, current_opt_state)
                new_params = optax.apply_updates(current_params, updates)
                return new_params, new_opt_state, loss
            
            # Run the optimization loop    
            for step in range(numberOfSteps):

                key, subkey = jax.random.split(key)
                batchOrder = jax.random.permutation(subkey, numberOfBatches)
                totalLoss = 0
                
                for i in range(numberOfBatches):
                    batch = dataset.getBatch(batchOrder[i])
                    self.params, optimizerState, loss = updateStep(self.params, optimizerState, batch.data)
                    totalLoss += loss

                if (step + 1) % 10 == 0:
                    print(f"Step {step+1:4d}, Loss: {totalLoss:.4f}, Number of Batches: {numberOfBatches}")
        
        else:

            if parameterUpdateFrequency == -1:
                parameterUpdateFrequency = numberOfBatches

            @jax.jit
            def calcGrad(params, batch):
                loss, grads = jax.value_and_grad(self.loss)(params, batch)
                return loss, grads

            @jax.jit
            def applyUpdate(grads, params, optState):
                updates, newOptState = optimizer.update(grads, optState)
                newParams = optax.apply_updates(params, updates)
                return newParams, newOptState
            
            # Run the optimization loop 
            for step in range(numberOfSteps):

                key, subkey = jax.random.split(key)
                batchOrder = jax.random.permutation(subkey, numberOfBatches)
                
                totalLoss = 0
                aggregatedGrads = jax.tree_util.tree_map(jax.numpy.zeros_like, self.params)
                aggregatedCount = 0

                for i in range(numberOfBatches):
                    batch = dataset.getBatch(batchOrder[i])
                    
                    loss, grads = calcGrad(self.params, batch.data)
                    totalLoss += loss
                    aggregatedGrads = jax.tree_util.tree_map(jax.numpy.add, aggregatedGrads, grads)
                    aggregatedCount += 1
                    
                    if aggregatedCount == parameterUpdateFrequency:
                        avgGrads = jax.tree_util.tree_map(lambda g: g / aggregatedCount, aggregatedGrads)
                        self.params, optimizerState = applyUpdate(avgGrads, self.params, optimizerState)                        
                        aggregatedGrads = jax.tree_util.tree_map(jax.numpy.zeros_like, self.params)
                        aggregatedCount = 0

                if aggregatedCount > 0:
                    avgGrads = jax.tree_util.tree_map(lambda g: g / aggregatedCount, aggregatedGrads)
                    self.params, optimizerState = applyUpdate(avgGrads, self.params, optimizerState)          

                if (step + 1) % 10 == 0:
                    print(f"Epoch {step+1:4d}, Avg Loss: {totalLoss:.4f}")

    def loss(self, params, data):
        return -self.logLikelihood(params, data)
    
    def hessian(self, params, rawBatch):
        # 1. Flatten the parameter pytree into a single 1D vector
        flatParams, pytreeDef = jax.tree_util.tree_flatten(params)
        raveledParams = jax.numpy.concatenate([p.ravel() for p in flatParams])
        
        # Helper info for unflattening
        paramShapes = [p.shape for p in flatParams]
        paramSizes = [p.size for p in flatParams]

        # 2. Create a wrapper loss function that accepts the flat vector
        @jax.jit
        def flatLossFn(flatParamsVec):
            # Un-flatten the vector back into the original pytree structure
            splitPoints = np.cumsum(paramSizes)[:-1]
            paramChunks = jax.numpy.split(flatParamsVec, splitPoints)
            unflattenedLeaves = [chunk.reshape(shape) for chunk, shape in zip(paramChunks, paramShapes)]
            unflattenedParams = jax.tree_util.tree_unflatten(pytreeDef, unflattenedLeaves)
            
            # Call the model-specific loss function
            return self.loss(unflattenedParams, rawBatch)
        
        # 3. Compute the Hessian
        return jax.hessian(flatLossFn)(raveledParams)

    def stdError(self, params, batch):        
        # Calculate the Hessian of the mean log-likelihood
        hessianOfMean = self.hessian(params, batch.data)
        
        # Scale to get the Hessian of the sum (correct for statistical inference)
        hessianOfSum = hessianOfMean * batch.getEffectiveNObs()
        
        try:
            # The variance-covariance matrix is the inverse of the Hessian
            covMatrix = jax.numpy.linalg.inv(hessianOfSum)
            stdErrorsFlat = jax.numpy.sqrt(jax.numpy.diag(covMatrix))
            
            # Un-flatten the std errors back into the original parameter shape
            flatParams, pytreeDef = jax.tree_util.tree_flatten(params)
            paramShapes = [p.shape for p in flatParams]
            paramSizes = [p.size for p in flatParams]
            
            splitPoints = np.cumsum(paramSizes)[:-1]
            stdErrorChunks = jax.numpy.split(stdErrorsFlat, splitPoints)
            unflattenedStdErrorLeaves = [chunk.reshape(shape) for chunk, shape in zip(stdErrorChunks, paramShapes)]
            
            return jax.tree_util.tree_unflatten(pytreeDef, unflattenedStdErrorLeaves)
            
        except np.linalg.LinAlgError:
            print("Hessian is not invertible. Standard errors cannot be computed.")
            return None
        
    def summary(self, batch, trueParams=None):
        print("\n--- Model Summary ---")
        stdErrors = self.stdError(self.params, batch)
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
        for key in self.params:
            param_array = self.params[key]
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