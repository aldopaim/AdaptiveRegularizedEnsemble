package moa.classifiers.meta;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.AbstractMOAObject;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.trees.ARFHoeffdingTree;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.core.Utils;
import moa.options.ClassOption;

public class AdaptiveRegularizedEnsemble extends AbstractClassifier implements MultiClassClassifier,
CapabilitiesHandler {

	@Override
	public String getPurposeString() {
		return "Adaptive regularized algorithm for evolving data streams from Paim and Enembreck.";
	}

	private static final long serialVersionUID = 1L;

	public ClassOption treeLearnerOption = new ClassOption("treeLearner", 'l',
			"Random Forest Tree.", ARFHoeffdingTree.class,
			"ARFHoeffdingTree -e 2000000 -g 100 -c 0.01");

	public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
			"The number of trees.", 100, 1, Integer.MAX_VALUE);

	
	public FloatOption lambdaOption = new FloatOption("lambda", 'a',
			"The lambda parameter for bagging.", 6.0, 1.0, Float.MAX_VALUE);

	public IntOption numberOfJobsOption = new IntOption("numberOfJobs", 'j',
			"Total number of concurrent jobs used for processing (-1 = as much as possible, 0 = do not use multithreading)", 1, -1, Integer.MAX_VALUE);

	public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
			"Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-3");

	public FlagOption disableDriftDetectionOption = new FlagOption("disableDriftDetection", 'u',
			"Should use drift detection?");

	public IntOption windowObservationSize  = new IntOption("windowObservationSize", 'w',
            "Size of the observation window to refine statistics and select learners in the voting.", 500, 0, Integer.MAX_VALUE);
	
		
	protected static final int SINGLE_THREAD = 0;

	protected AREBaseLearner[] ensemble;
	protected long instancesSeen;
	protected int subspaceSize;
	
	private ExecutorService executor;

	//statistic window size
    protected double avgAccuracyWindowLearner;
    
  //subspace
    protected Random subspaceRandom;
    protected int  maxValueRandom;
    protected int  minValueRandom;
    
    
	@Override
	public void resetLearningImpl() {
		// Reset attributes
		this.ensemble = null;
		this.subspaceSize = 0;
		this.instancesSeen = 0;
		this.avgAccuracyWindowLearner = 0;

		// Multi-threading
		int numberOfJobs;
		if(this.numberOfJobsOption.getValue() == -1) 
			numberOfJobs = Runtime.getRuntime().availableProcessors();
		else 
			numberOfJobs = this.numberOfJobsOption.getValue();
		// SINGLE_THREAD and requesting for only 1 thread are equivalent. 
		// this.executor will be null and not used...
		if(numberOfJobs != AdaptiveRegularizedEnsemble.SINGLE_THREAD && numberOfJobs != 1)
			this.executor = Executors.newFixedThreadPool(numberOfJobs);
		
	}

	@Override
	public void trainOnInstanceImpl(Instance instance) {
		++this.instancesSeen;
		if(this.ensemble == null) 
			initEnsemble(instance);
		
		double accWindowLearner = 0.0;
		double sumOfSquares = 0.0;
		
		Collection<TrainingRunnable> trainers = new ArrayList<TrainingRunnable>();
		for (int i = 0 ; i < this.ensemble.length ; i++) {
			
			long seed = Double.toString(instancesSeen).hashCode()+i;
            this.ensemble[i].setSeedRandom(seed);

			DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(instance));
			
			int trueClass = (int) instance.classValue();
            int predictedClass = Utils.maxIndex(vote.getArrayRef());
			
            /*To avoid that domains with high noise incidence fully participate in the model creation. 
             * We use the strategy that for every five rejections, 
             * one instance is trained even though it is classified correctly*/
            boolean willTrain = trueClass != predictedClass;
            
            
            //hit
            if (!willTrain) {
            	this.ensemble[i].UntrainedClasses[trueClass]++;
            	if (this.ensemble[i].UntrainedClasses[trueClass] >= 5) {
            		this.ensemble[i].UntrainedClasses[trueClass] = 0;
            		willTrain = true;
            	}
            }
            
            if (willTrain) {
	            int k = MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
				if (k > 0) {
					if(this.executor != null) {
						TrainingRunnable trainer = new TrainingRunnable(this.ensemble[i], 
								instance, k, this.instancesSeen);
						trainers.add(trainer);
					}
					else { // SINGLE_THREAD is in-place... 
						this.ensemble[i].trainOnInstance(instance, k, this.instancesSeen);
					}
				}
			}

            
            this.ensemble[i].updateMatrixConfusion(trueClass == predictedClass, 1);
			accWindowLearner += this.ensemble[i].accuracyWindowLearner;
			sumOfSquares += (this.ensemble[i].accuracyWindowLearner * this.ensemble[i].accuracyWindowLearner);
		}
		
		
		if (accWindowLearner > 0.0) {
			avgAccuracyWindowLearner = (accWindowLearner / this.ensemble.length);

			double aux = ((accWindowLearner*accWindowLearner)/this.ensemble.length);
			double stDev = Math.sqrt( (sumOfSquares - aux ) / (this.ensemble.length-1)  );
			avgAccuracyWindowLearner = avgAccuracyWindowLearner - stDev;
		}
		
		
		if(this.executor != null) {
			try {
				this.executor.invokeAll(trainers);
			} catch (InterruptedException ex) {
				throw new RuntimeException("Could not call invokeAll() on training threads.");
			}
		}
		
	}

	@Override
	public double[] getVotesForInstance(Instance instance) {
		Instance testInstance = instance.copy();
		if(this.ensemble == null) 
			initEnsemble(testInstance);
		DoubleVector combinedVote = new DoubleVector();
		for(int i = 0 ; i < this.ensemble.length ; ++i) {
			boolean shouldClassifierVote = (this.ensemble[i].accuracyWindowLearner >= avgAccuracyWindowLearner);
			//boolean shouldClassifierVote = true;
			if (shouldClassifierVote) {
				DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(testInstance));
				
				 if (vote.sumOfValues() > 0.0) { 
		                vote.normalize();
		                combinedVote.addValues(vote);
		        }
			}
		}
		
		return combinedVote.getArrayRef();
	}

	@Override
	public boolean isRandomizable() {
		return true;
	}

	@Override
	public void getModelDescription(StringBuilder arg0, int arg1) {
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	protected void initEnsemble(Instance instance) {
		// Init the ensemble.
		int ensembleSize = this.ensembleSizeOption.getValue();
		this.ensemble = new AREBaseLearner[ensembleSize];

		int n = instance.numAttributes()-1; // Ignore class label ( -1 )
		
		this.subspaceRandom = new Random();
		this.subspaceRandom.setSeed(n+instance.numClasses());
		this.minValueRandom = 2; 
		this.maxValueRandom = n;
		
		
		ARFHoeffdingTree treeLearner = (ARFHoeffdingTree) getPreparedClassOption(this.treeLearnerOption);
		treeLearner.resetLearning();

		for(int i = 0 ; i < ensembleSize ; ++i) {
			
			treeLearner.subspaceSizeOption.setValue(randomSubSpaceSize());
			
			this.ensemble[i] = new AREBaseLearner(
					i, 
					(ARFHoeffdingTree) treeLearner.copy(), 
					//(BasicClassificationPerformanceEvaluator) classificationEvaluator.copy(), 
					this.instancesSeen, 
					! this.disableDriftDetectionOption.isSet(), 
					driftDetectionMethodOption,
					this.windowObservationSize.getValue());
			
			this.ensemble[i].UntrainedClasses = new long[instance.dataset().numClasses()];

		}
	}

	private int randomSubSpaceSize() {
		
		int randomSubSpaceSize = this.subspaceRandom.nextInt(maxValueRandom + 1 - minValueRandom) + minValueRandom;
		return randomSubSpaceSize;
	}

	@Override
	public ImmutableCapabilities defineImmutableCapabilities() {
		if (this.getClass() == AdaptiveRegularizedEnsemble.class)
			return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
		else
			return new ImmutableCapabilities(Capability.VIEW_STANDARD);
	}

	@Override
	public Classifier[] getSublearners() {
		/* Extracts the reference to the ARFHoeffdingTree object from within the ensemble of AREBaseLearner's */
		Classifier[] forest = new Classifier[this.ensemble.length];
		for(int i = 0 ; i < forest.length ; ++i)
			forest[i] = this.ensemble[i].classifier;
		return forest;

	}

	/**
	 * Inner class that represents a single tree member of the forest. 
	 */
	protected final class AREBaseLearner extends AbstractMOAObject {
		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;
		
		public int indexOriginal;
		public long createdOn;
		public long lastDriftOn;
		public long lastWarningOn;
		public ARFHoeffdingTree classifier;

		protected ClassOption driftOption;
		protected ChangeDetector driftDetectionMethod;
		public boolean useDriftDetector;

		protected int numberOfDriftsDetected;
		protected int numberOfWarningsDetected;
		
		//
		 //protected int windowObservationSize = 400;
		 protected int windowObservationSize;
		 protected double accuracyWindowLearner; 

		 protected int[] accClassifierArray; 
		 protected int lastIndex;
		 protected Random subspaceRandomBaseLearner;
		 
		 protected long[] UntrainedClasses;

		private void init(int indexOriginal, 
				ARFHoeffdingTree instantiatedClassifier, 
				long instancesSeen, 
				boolean useDriftDetector, 
				ClassOption driftOption,
				int windowObservationSize) {
			this.indexOriginal = indexOriginal;
			this.createdOn = instancesSeen;
			this.lastDriftOn = 0;
			this.lastWarningOn = 0;

			this.classifier = instantiatedClassifier;
			this.useDriftDetector = useDriftDetector;

			this.numberOfDriftsDetected = 0;
			this.numberOfWarningsDetected = 0;

			if(this.useDriftDetector) {
				this.driftOption = driftOption;
				this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();
			}

			this.windowObservationSize = windowObservationSize;
		}

		public AREBaseLearner(int indexOriginal, 
				ARFHoeffdingTree instantiatedClassifier, 
				long instancesSeen, 
				boolean useDriftDetector, 
				ClassOption driftOption,
				int windowObservationSize) {
			init(indexOriginal, 
					instantiatedClassifier, 
					instancesSeen, 
					useDriftDetector, 
					driftOption,
					windowObservationSize);
		}

		public void reset() {
			
			this.classifier.resetLearning();
			this.createdOn = instancesSeen;
			this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();
			
			accClassifierArray = null;
            this.classifier.subspaceSizeOption.setValue(randomSubSpaceSizeLocal());
		}
		
		public void setSeedRandom(long seed) {
			if (this.subspaceRandomBaseLearner == null)
				this.subspaceRandomBaseLearner = new Random();
			this.subspaceRandomBaseLearner.setSeed(seed);
		}

		private int randomSubSpaceSizeLocal() {
			int randomSubSpaceSize = this.subspaceRandomBaseLearner.nextInt(maxValueRandom + 1 - minValueRandom) + minValueRandom;
			return randomSubSpaceSize;
		}
	        

		public void trainOnInstance(Instance instance, double weight, long instancesSeen) {
			Instance weightedInstance = instance.copy();
			weightedInstance.setWeight(instance.weight() * weight);
			this.classifier.trainOnInstance(weightedInstance);


			boolean correctlyClassifies = this.classifier.correctlyClassifies(instance);

			/*********** drift detection ***********/
			if(this.useDriftDetector) {
				// Update the DRIFT detection method
				this.driftDetectionMethod.input(correctlyClassifies ? 0 : 1);
				// Check if there was a change
				if(this.driftDetectionMethod.getChange()) {
					this.lastDriftOn = instancesSeen;
					this.numberOfDriftsDetected++;
					this.reset();
				}
			}
		}

		private void updateMatrixConfusion(boolean correctlyClassifies, double weight) {
        	double acc = 0.0;
            
            if (accClassifierArray == null) {
            	accClassifierArray = new int[this.windowObservationSize+2];
            	lastIndex = -1;
            }
            
            if (lastIndex == this.windowObservationSize-1) {
            	lastIndex = -1;
            	
            }
            	
            if (accClassifierArray[accClassifierArray.length-2] < this.windowObservationSize) {
            	accClassifierArray[accClassifierArray.length-2] ++;
            }
            
            lastIndex++;
            accClassifierArray[accClassifierArray.length-1] -= 
            		accClassifierArray[lastIndex];
            accClassifierArray[lastIndex] = correctlyClassifies ? (int)weight : 0;
            
            
            accClassifierArray[accClassifierArray.length-1] += 
            		accClassifierArray[lastIndex];
            
            acc = (double) accClassifierArray[accClassifierArray.length-1]/accClassifierArray[accClassifierArray.length-2];
            accuracyWindowLearner = acc;
			
		}
		
		public double[] getVotesForInstance(Instance instance) {
			DoubleVector vote = new DoubleVector(this.classifier.getVotesForInstance(instance));
			return vote.getArrayRef();
		}

		@Override
		public void getDescription(StringBuilder sb, int indent) {
		}
	}

	/***
	 * Inner class to assist with the multi-thread execution. 
	 */
	protected class TrainingRunnable implements Runnable, Callable<Integer> {
		final private AREBaseLearner learner;
		final private Instance instance;
		final private double weight;
		final private long instancesSeen;

		public TrainingRunnable(AREBaseLearner learner, Instance instance, 
				double weight, long instancesSeen) {
			this.learner = learner;
			this.instance = instance;
			this.weight = weight;
			this.instancesSeen = instancesSeen;
		}

		@Override
		public void run() {
			learner.trainOnInstance(this.instance, this.weight, this.instancesSeen);
		}

		@Override
		public Integer call() {
			run();
			return 0;
		}
	}
}
