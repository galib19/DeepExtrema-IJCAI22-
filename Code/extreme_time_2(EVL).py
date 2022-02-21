import pickle
import time

import tensorflow as tf
import numpy as np

from ts.utility import Utility
from ts.log import GlobalLogger, ConsoleLogger


class ExtremeTime2:

    @staticmethod
    def load(modelLoadPath):
        """
        Loads the model from the provided filepath
        :param modelLoadPath: path from where to load the model
        :return: model which is loaded from the given path
        """

        logger = GlobalLogger.getLogger()

        model = ExtremeTime2(loadModel=True)

        with open(modelLoadPath, 'rb') as fl:
            logger.log(
                'Load Dictionary from Model Params file',
                1, ExtremeTime2.load.__name__
            )
            loadDict = pickle.load(fl)

        logger.log('Loading Params', 1, ExtremeTime2.load.__name__)

        model.forecastHorizon = loadDict['forecastHorizon']
        model.memorySize = loadDict['memorySize']
        model.windowSize = loadDict['windowSize']
        model.inputDimension = loadDict['inputDimension']
        model.embeddingSize = loadDict['embeddingSize']
        model.contextSize = loadDict['contextSize']
        model.memory = loadDict['memory']
        model.context = loadDict['context']

        model.buildModel()
        model.gruInput.set_weights(loadDict['gruInput'])
        model.gruMemory.set_weights(loadDict['gruMemory'])
        model.gruContext.set_weights(loadDict['gruContext'])
        model.outDense.set_weights(loadDict['outDense'])
        model.state = loadDict['state']

        return model

    def __init__(
            self,
            forecastHorizon=1,
            memorySize=80,
            windowSize=60,
            embeddingSize=10,
            contextSize=10,
            numExoVariables=0,
            loadModel=False
    ):
        """
        Initialize the model parameters and hyperparameters
        :param forecastHorizon: How much further in the future the model has to
        predict the target series variable
        :param memorySize: Size of the explicit memory unit used by the model, it
        should be a scalar value
        :param windowSize: Size of each window which is to be compressed and stored
        as a memory cell
        :param embeddingSize: Size of the hidden state of the GRU encoder
        :param contextSize: Size of context produced from historical sequences
        :param numExoVariables: Number of exogenous variables the model takes as input
        :param loadModel: True or False - do not use this parameter !,
        this is for internal use only (i.e. it is an implementation detail)
        If True, then object is normally created, else object is created
        without any member values being created. This is used when model
        is created by the static load method
        """

        tf.keras.backend.set_floatx('float64')

        if loadModel:
            return

        logger = GlobalLogger.getLogger()
        logger.log('Initializing Members', 1, self.__init__.__name__)

        self.forecastHorizon = forecastHorizon
        self.memorySize = memorySize
        self.windowSize = windowSize
        self.embeddingSize = embeddingSize
        self.contextSize = contextSize
        self.inputDimension = numExoVariables + 1
        self.memory = None
        self.context = None

        logger.log('Building Model Parameters', 1, self.__init__.__name__)

        self.outDense = self.gruContext = self.gruMemory = self.gruInput = None
        self.buildModel()
        self.state = self.getInitialState()

        logger.log(f'state shape: {self.state.shape}', 2, self.predict.__name__)

    def train(
            self,
            targetSeries,
            sequenceLength,
            exogenousSeries=None,
            numIterations=1,
            optimizer=tf.optimizers.Adam(),
            modelSavePath=None,
            verboseLevel=1,
            returnLosses=True
    ):
        """
        Train the Model Parameters on the provided data
        :param targetSeries: Univariate Series of the Target Variable, it
        should be a numpy array of shape (n + self.forecastHorizon,)
        :param sequenceLength: Length of each training sequence
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (n, numExoVariables), it can be None only if
        numExoVariables is 0 in which case the exogenous variables are not
        considered
        :param numIterations: Number of iterations of training to be performed
        :return: If returnLosses is True, then numpy array of losses of shape (numSeq,)
        :param optimizer: Optimizer of training the parameters
        :param modelSavePath: Path where to save the model parameters after
        each training an a sequence, if None then parameters are not saved
        :param verboseLevel: Verbose level, 0 is nothing, greater values increases
        the information printed to the console
        :param returnLosses: If True, then losses are returned, else losses are not
        returned
        is returned, else None is returned
        """

        logger = GlobalLogger.getLogger()
        verbose = ConsoleLogger(verboseLevel)

        assert (Utility.isExoShapeValid(exogenousSeries, self.inputDimension - 1))
        X, Y = Utility.prepareDataTrain(targetSeries, exogenousSeries, self.forecastHorizon)

        n = X.shape[0]
        logger.log(f'Seq Start Time: {self.windowSize}, Train len: {n}', 2, self.train.__name__)
        assert (self.windowSize < n)

        logger.log('Begin Training', 1, self.train.__name__)

        losses = []

        for iteration in range(numIterations):

            verbose.log(f'begin iteration {iteration}', 1)

            seqStartTime = self.windowSize
            cumulIterLoss = 0.0
            numSeq = 0

            iterStartTime = time.time()
            while seqStartTime < n:
                seqEndTime = min(seqStartTime + sequenceLength, n - 1)

                startTime = time.time()
                loss = self.trainSequence(X, Y, seqStartTime, seqEndTime, optimizer)
                endTime = time.time()
                timeTaken = endTime - startTime

                cumulIterLoss += loss
                numSeq += 1

                verbose.log(f'start timestep: {seqStartTime}'
                            + f' | end timestep: {seqEndTime}'
                            + f' | time taken: {timeTaken : .2f} sec'
                            + f' | Loss: {loss}', 2)

                seqStartTime += sequenceLength

            iterEndTime = time.time()
            iterTimeTaken = iterEndTime - iterStartTime
            avgIterLoss = cumulIterLoss / numSeq

            verbose.log(f'Completed Iteration: {iteration}'
                        + f' | time taken: {iterTimeTaken : .2f} sec'
                        + f' | Avg Iteration Loss: {avgIterLoss}', 1)

            if returnLosses:
                losses.append(avgIterLoss)

            if modelSavePath is not None:
                logger.log(f'Saving Model at {modelSavePath}', 1, self.train.__name__)
                self.save(modelSavePath)

        self.buildMemory(X, n)

        if returnLosses:
            return np.array(losses)

    def predict(
            self,
            targetSeries,
            exogenousSeries=None
    ):
        """
        Forecast using the model parameters on the provided input data
        :param targetSeries: Univariate Series of the Target Variable, it
        should be a numpy array of shape (n,)
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (n, numExoVariables), it can be None only if
        numExoVariables is 0 in which case the exogenous variables are not
        considered
        :return: Forecast targets predicted by the model, it has shape (n,), the
        horizon of the targets is the same as self.forecastHorizon
        """

        logger = GlobalLogger.getLogger()
        logger.log('Begin Prediction', 1, self.predict.__name__)

        assert (Utility.isExoShapeValid(exogenousSeries, self.inputDimension - 1))
        X = Utility.prepareDataPred(targetSeries, exogenousSeries)

        n = X.shape[0]
        Ypred = [None] * n

        for t in range(n):
            Ypred[t] = self.predictTimestep(X, t)

        Ypred = np.array(Ypred)
        logger.log(f'Output Shape: {Ypred.shape}', 2, self.predict.__name__)

        return Ypred

    def evaluate(
            self,
            targetSeries,
            exogenousSeries=None,
            returnPred=False
    ):
        """
        Forecast using the model parameters on the provided data, evaluates
        the forecast result using the loss and returns it
        :param targetSeries: Univariate Series of the Target Variable, it
        should be a numpy array of shape (numTimesteps + self.forecastHorizon,).
        numTimesteps is the number of timesteps on which our model must predict,
        the values ahead are for evaluating the predicted results with respect
        to them (i.e. they are true targets for our prediction)
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (numTimesteps, numExoVariables), it can be None
        only if numExoVariables is 0 in which case the exogenous variables
        are not considered
        :param returnPred: If True, then return predictions along with loss, else
        return on loss
        :return: If True, then return predictions along with loss of the predicted
        and true targets, else return only loss
        """

        logger = GlobalLogger.getLogger()
        logger.log('Begin Evaluating', 1, self.evaluate.__name__)

        n = targetSeries.shape[0] - self.forecastHorizon
        logger.log(f'Evaluate Sequence Length: {n}', 2, self.evaluate.__name__)
        assert (n >= 0)

        if exogenousSeries is not None:
            logger.log(
                f'Exogenous Series Shape: {exogenousSeries.shape}',
                2,
                self.evaluate.__name__
            )
            assert (exogenousSeries.shape[0] == n)

        Ypred = self.predict(targetSeries[:n], exogenousSeries)

        loss = tf.keras.losses.MSE(targetSeries[self.forecastHorizon:], Ypred)

        logger.log(f'Computed Loss: {loss}', 2, self.evaluate.__name__)

        if returnPred:
            return loss, Ypred
        else:
            return loss

    def save(
            self,
            modelSavePath
    ):
        """
        Save the model parameters at the provided path
        :param modelSavePath: Path where the parameters are to be saved
        :return: None
        """

        logger = GlobalLogger.getLogger()

        assert (self.memory is not None)
        logger.log(f'Memory Shape: {self.memory.shape}', 2, self.save.__name__)

        logger.log('Constructing Dictionary from model params', 1, self.save.__name__)

        saveDict = {
            'forecastHorizon': self.forecastHorizon,
            'memorySize': self.memorySize,
            'windowSize': self.windowSize,
            'inputDimension': self.inputDimension,
            'embeddingSize': self.embeddingSize,
            'contextSize': self.contextSize,
            'memory': self.memory,
            'context': self.context,
            'gruInput': self.gruInput.get_weights(),
            'gruMemory': self.gruMemory.get_weights(),
            'gruContext': self.gruContext.get_weights(),
            'outDense': self.outDense.get_weights(),
            'state': self.state
        }

        logger.log('Saving Dictionary', 1, self.save.__name__)

        fl = open(modelSavePath, 'wb')
        pickle.dump(saveDict, fl)
        fl.close()

    def trainSequence(self, X, Y, seqStartTime, seqEndTime, optimizer):
        """
        :param X: Features, has shape (n, self.inputShape)
        :param Y: Targets, has shape (n,)
        :param seqStartTime: Sequence Start Time
        :param seqEndTime: Sequence End Time
        :param optimizer: The optimization algorithm
        :return: The loss value resulted from training on the sequence
        """

        logger = GlobalLogger.getLogger()
        logger.log('Begin Training on Sequence', 1, self.trainSequence.__name__)
        logger.log(f'Sequence start: {seqStartTime}, Sequence end: {seqEndTime}', 2, self.trainSequence.__name__)

        with tf.GradientTape() as tape:
            self.buildMemory(X, seqStartTime)

            Ypred = []
            for t in range(seqStartTime, seqEndTime + 1):
                pred = self.predictTimestep(X, t)
                Ypred.append(pred)

            Ypred = tf.convert_to_tensor(Ypred, dtype=tf.float64)
            logger.log(f'Prediction Shape: {Ypred.shape}', 2, self.trainSequence.__name__)

            loss = tf.keras.losses.MSE(
                Y[seqStartTime: seqEndTime + 1],
                Ypred
            )
            logger.log(f'Loss: {loss}', 2, self.trainSequence.__name__)

        trainableVars = \
            self.gruInput.trainable_variables \
            + self.gruMemory.trainable_variables \
            + self.gruContext.trainable_variables \
            + self.outDense.trainable_variables

        logger.log('Performing Gradient Descent', 1, self.trainSequence.__name__)

        grads = tape.gradient(loss, trainableVars)
        assert (len(trainableVars) == len(grads))

        optimizer.apply_gradients(zip(
            grads,
            trainableVars
        ))

        return loss

    def buildMemory(self, X, currentTime):
        """
        Build Model Memory using the timesteps seen up till now
        :param X: Features, has shape (n, self.inputShape)
        :param currentTime: current timestep, memory would be built only using the
        timestep earlier than the current timestep
        :return: None
        """

        logger = GlobalLogger.getLogger()
        logger.log(f'Building Memory', 1, self.buildMemory.__name__)
        logger.log(f'Current Time: {currentTime}', 2, self.buildMemory.__name__)
        assert (currentTime >= self.windowSize)

        sampleLow = 0
        sampleHigh = currentTime - self.windowSize

        self.memory = [None] * self.memorySize
        self.context = [None] * self.memorySize

        for i in range(self.memorySize):
            windowStartTime = np.random.randint(
                sampleLow,
                sampleHigh + 1
            )

            self.memory[i], self.context[i] = self.runGruOnWindow(X, windowStartTime)

        self.memory = tf.stack(self.memory)
        self.context = tf.stack(self.context)

        logger.log(
            f'Memory Shape: {self.memory.shape}, Context Shape: {self.context.shape}',
            2,
            self.buildMemory.__name__
        )

    def runGruOnWindow(self, X, windowStartTime):
        """
        Runs GRU on the window and returns the final state
        :param X: Features, has shape (n, self.inputShape)
        :param windowStartTime: Starting timestep of the window
        :return: The final state after running on the window, it has shape (self.encoderStateSize,)
        """

        logger = GlobalLogger.getLogger()
        logger.log(f'Window Start Time: {windowStartTime}', 2, self.runGruOnWindow.__name__)

        gruMemoryState, gruContextState = self.getInitialEncoderStates()

        for t in range(
                windowStartTime,
                windowStartTime + self.windowSize
        ):
            gruMemoryState, _ = self.gruMemory(
                np.expand_dims(X[t], 0),
                gruMemoryState
            )

            gruContextState, _ = self.gruContext(
                np.expand_dims(X[t], 0),
                gruContextState
            )

        gruMemoryState = tf.squeeze(gruMemoryState, axis=0)
        gruContextState = tf.squeeze(gruContextState, axis=0)

        logger.log(
            f'GRU memory state shape: {gruMemoryState.shape},'
            + f' context state shape: {gruContextState.shape}',
            2,
            self.runGruOnWindow.__name__
        )

        return gruMemoryState, gruContextState

    def predictTimestep(self, X, currentTime):
        """
        Predict on a Single Timestep
        :param X: Features, has shape (n, self.inputShape)
        :param currentTime: Current Timestep
        :return: The predicted value on current timestep and the next state
        """

        logger = GlobalLogger.getLogger()

        self.state, _ = self.gruInput(
            np.expand_dims(X[currentTime], axis=0),
            self.state
        )

        embedding = tf.squeeze(self.state)

        attentionWeights = self.computeAttention(embedding)
        logger.log(f'Attention Shape: {attentionWeights.shape}', 2, self.predictTimestep.__name__)

        weightedContext = \
            tf.expand_dims(attentionWeights, axis=1) * self.context

        concatVector = tf.concat([
            embedding,
            tf.reshape(weightedContext, (tf.size(weightedContext),))
        ], axis=0)

        logger.log(f'Concat Vector Shape: {concatVector.shape}', 2, self.predictTimestep.__name__)

        pred = tf.squeeze(self.outDense(tf.expand_dims(concatVector, axis=0)))
        logger.log(f'Prediction: {pred}', 2, self.predictTimestep.__name__)

        return pred

    def computeAttention(self, embedding):
        """
        Computes Attention Weights by taking softmax of the inner product
        between embedding of the input and the memory states
        :param embedding: Embedding of the input, it has shape (self.embeddingSize,)
        :return: Attention Weight Values
        """

        return tf.nn.softmax(tf.squeeze(tf.linalg.matmul(
            self.memory,
            tf.expand_dims(embedding, axis=1)
        )))

    def buildModel(self):
        """ Build Model Architecture """

        self.gruInput = tf.keras.layers.GRUCell(self.embeddingSize)
        self.gruInput.build(input_shape=(self.inputDimension,))

        self.gruMemory = tf.keras.layers.GRUCell(self.embeddingSize)
        self.gruMemory.build(input_shape=(self.inputDimension,))

        self.gruContext = tf.keras.layers.GRUCell(self.contextSize)
        self.gruContext.build(input_shape=(self.inputDimension,))

        finalWeightSize = self.embeddingSize + self.contextSize * self.memorySize
        self.outDense = tf.keras.layers.Dense(1)
        self.outDense.build(input_shape=(finalWeightSize,))

    def getInitialState(self):
        """
        Computes Initial Input GRU's State
        :return: Initial Input GRU's State
        """

        return self.gruInput.get_initial_state(
            batch_size=1,
            dtype=tf.float64
        )

    def getInitialEncoderStates(self):
        """
        Computes Initial GRU Encoder State
        :return: Initial GRU State
        """

        return \
            self.gruMemory.get_initial_state(
                batch_size=1,
                dtype=tf.float64
            ), self.gruContext.get_initial_state(
                batch_size=1,
                dtype=tf.float64
            )