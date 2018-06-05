package eus.ehu.tss;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Main class of the library that will conduct the classification of the images.
 */
public class TrainableSuperpixelSegmentation {

    private ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList<RegionFeatures.Feature>();
    private ImagePlus inputImage;
    private ImagePlus labelImage;
    private ImagePlus resultImage;
    private Instances unlabeled;
    private Instances labeled;
    private Instances trainingData;
    private AbstractClassifier abstractClassifier;
    private boolean classifierTrained = false;
    ArrayList<String> classes = null;

    /**
     * Empty builder
     */
    public TrainableSuperpixelSegmentation(){

    }


    /**
     * Creates instance of TrainableSuperpixelSegmentation based on an image and it's corresponding label image, a list of selected features and an AbstractClassifier
     * @param originalImage ImagePlus image that will be analyzed
     * @param labels ImagePlus labeled image of the originalImage
     * @param features ArrayList of Features (from RegionFeatures.Feature) that represent the features that will be calculated
     * @param classifier AbstractClassifier that will be used to classify the images
     */
    public TrainableSuperpixelSegmentation(
    		ImagePlus originalImage,
    		ImagePlus labels,
    		ArrayList<RegionFeatures.Feature> features,
    		AbstractClassifier classifier,
    		ArrayList<String> classes )
    {
        selectedFeatures = features;
        inputImage = originalImage;
        labelImage = labels;
        abstractClassifier  = classifier;
        this.classes = classes;
    }

    /**
     * Calculates the selected features for each region and saves them on the private variable unlabeled
     * @return boolean that checks if the region features have been created
     */
    public boolean calculateRegionFeatures(){
        if(inputImage.getType()==ImagePlus.COLOR_RGB){
            unlabeled = RegionColorFeatures.calculateUnlabeledColorFeatures(
                    inputImage,
                    labelImage,
                    selectedFeatures,
                    classes);
        }else {
            unlabeled = RegionFeatures.calculateUnlabeledRegionFeatures(
                    inputImage,
                    labelImage,
                    selectedFeatures,
                    classes);
        }
        return unlabeled != null;
    }


    /**
     * Returns a String with ARFF format of the features for each region
     * @return String in ARFF format
     */
    public String getFeaturesByRegion(){
        return unlabeled.toString();
    }

    /**
     * Trains classifiers based on previously created features and a list of classes with their corresponding regions
     * @param classRegions ArrayList of int[] where each int[] represents the labels of superpixels that belong to the class indicated by their index in the ArrayList
     * @return boolean value false when training has had an error
     */
    public boolean trainClassifier(ArrayList<int[]> classRegions){
    	// read attributes from unlabeled data
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        int numFeatures = unlabeled.numAttributes()-1;
        for(int i=0;i<numFeatures;++i){
            attributes.add(new Attribute(unlabeled.attribute(i).name(),i));
        }
        attributes.add(new Attribute("Class",classes));
        trainingData = new Instances("training data",attributes,0);
        // Fill training dataset with the feature vectors of the corresponding
        // regions given by classRegions
        for(int i=0;i<classRegions.size();++i){ //For each class in classRegions
            for(int j=0;j<classRegions.get(i).length;++j){
                Instance inst = new DenseInstance(numFeatures+1);
                for(int k=0;k<numFeatures;++k){
                    inst.setValue(k,unlabeled.get(classRegions.get(i)[j]-1).value(k));
                }
                inst.setValue(numFeatures,i); // set class value
                trainingData.add(inst);
            }
        }
        trainingData.setClassIndex(numFeatures); // set class index

        try {
            abstractClassifier.buildClassifier(trainingData);
            classifierTrained = true;
            return true;
        } catch (Exception e) {
            System.out.println("Error when building classifier");
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Train classifier based on previously loaded training data
     * @return boolean value false when training has had an error
     */
    public boolean trainClassifier(){
        try {
            abstractClassifier.buildClassifier(trainingData);
            classifierTrained = true;
            return true;
        } catch (Exception e) {
            System.out.println("Error when building classifier");
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Applies classifier to unlabeled data and creates new Instances in labeled private variables
     * @return ImagePlus with classified image
     */
    public ImagePlus applyClassifier(){
        try {
            labeled = new Instances(unlabeled); //Copy of unlabeled to label
            for (int i = 0; i < unlabeled.numInstances(); ++i) {
                double classLabel = abstractClassifier.classifyInstance(unlabeled.instance(i));
                labeled.instance(i).setClassValue(classLabel);
            }
        } catch (Exception e) {
            System.out.println("Error when applying classifier");
            e.printStackTrace();
        }
        int height = inputImage.getHeight();
        int width = inputImage.getWidth();
        float tags[] = new float[height*width];
        ImageStack result = new ImageStack(width,height);
        ImageStack stackLabels = labelImage.getStack();
        for(int slice = 1; slice <= inputImage.getNSlices(); ++slice) {
            ImageProcessor ip = stackLabels.getProcessor(slice);
            for (int x = 0; x < width; ++x) {
                for (int y = 0; y < height; ++y) {
                    int index = ip.getPixel(x, y);
                    if (index == 0) { //edge pixel
                        tags[x + y * width] = index;
                    } else {
                        Instance instance = labeled.get(index - 1);
                        tags[x + y * width] = (float) instance.classValue();
                    }
                }
            }
            FloatProcessor processor = new FloatProcessor(width, height, tags);
            result.addSlice(stackLabels.getSliceLabel(slice),processor.duplicate());
        }
        resultImage = new ImagePlus(inputImage.getShortTitle()+"-supseg",result);
        return resultImage;
    }

    /**
     * Applies already trained classifier to input image and label image
     * @param inImage input image
     * @param lbImage superpixel segmentated label image
     * @return classified image
     */
    public ImagePlus applyClassifier(ImagePlus inImage, ImagePlus lbImage){
        if(abstractClassifier==null){
            System.out.println("Train a classifier first!");
            return inImage;
        }else{
            inputImage = inImage;
            labelImage = lbImage;
            if(calculateRegionFeatures()){
                return applyClassifier();
            }else {
                System.out.println("Error when calculating region features");
                return inImage;
            }
        }
    }

    /**
     * Returns a probability map image
     * @return
     */
    public ImagePlus getProbabilityMap(){
        final int numInstances = unlabeled.numInstances();
        final int numClasses = classes.size();
        final int width = labelImage.getWidth();
        final int height = labelImage.getHeight();
        final double[][] classificationResult = new double[numClasses][numInstances];
        ImageStack classificationResultImage = new ImageStack(width,height);
        for(int i=0;i<numInstances;++i){
            try{
                double[] prob = abstractClassifier.distributionForInstance(unlabeled.get(i));
                for(int k=0; k<numClasses;++k){
                    classificationResult[k][i] = prob[k];
                }
            }catch (Exception e){
                System.out.println("Could not apply Classifier!");
                e.printStackTrace();
                return null;
            }
        }
        ImageStack stackLabels = labelImage.getStack();
        double tags[] = new double[height*width];
        for(int k = 0;k<numClasses;++k) {
            for (int slice = 1; slice <= inputImage.getNSlices(); ++slice) {
                ImageProcessor ip = stackLabels.getProcessor(slice);
                for (int x = 0; x < width; ++x) {
                    for (int y = 0; y < height; ++y) {
                        int index = ip.getPixel(x, y);
                        if (index == 0) { //edge pixel
                            tags[x + y * width] = index;
                        } else {
                            tags[x + y * width] = classificationResult[k][index-1];
                        }
                    }
                }
                FloatProcessor processor = new FloatProcessor(width, height,tags);
                classificationResultImage.addSlice(classes.get(k),processor.duplicate());
            }
        }
        ImagePlus result = new ImagePlus(inputImage.getShortTitle()+"-probMap",classificationResultImage);
        return result;

    }

    /**
     * Set a classifier
     * @param classifier input classifier
     * @return true on success and false on failure
     */
    public boolean setClassifier(AbstractClassifier classifier){
        if(classifier!=null){
            abstractClassifier = classifier;
            return true;
        }
        return false;
    }

    /**
     * Returns classifier
     * @return classifier
     */
    public AbstractClassifier getClassifier(){
        return abstractClassifier;
    }

    /**
     * Returns labeled instances
     * @return
     */
    public Instances getInstances() {
        return labeled;
    }

    /**
     * Return result image
     * @return result image
     */
    public ImagePlus getResultImage() {
        return resultImage;
    }

    /**
     * Returns training data
     * @return training data
     */
    public Instances getTrainingData() {
        return trainingData;
    }


    /**
     * Returns list of classes
     * @return list of classes
     */
    public ArrayList<String> getClasses() {
        return classes;
    }

    /**
     * Sets list of classes
     * @param classes ArrayList of class names
     */
    public void setClasses(ArrayList<String> classes) {
        this.classes = classes;
    }


    /**
     * Sets selected features
     * @param selectedFeatures ArrayList of RegionFeatures.Feature elements
     */
    public void setSelectedFeatures(ArrayList<RegionFeatures.Feature> selectedFeatures) {
        this.selectedFeatures = selectedFeatures;
    }

    /**
     * Sets training data
     * @param trainingData Instances to be used in the training
     */
    public void setTrainingData(Instances trainingData) {
        this.trainingData = trainingData;
    }

    /**
     * Returns provided input image
     * @return ImagePlus with input image
     */
    public ImagePlus getInputImage() {
        return inputImage;
    }


    /**
     * Set input image
     * @param inputImage
     */
    public void setInputImage(ImagePlus inputImage) {
        this.inputImage = inputImage;
    }

    /**
     * Returns superpixel image with labels
     * @return
     */
    public ImagePlus getLabelImage() {
        return labelImage;
    }

    /**
     * Sets superpixel image
     * @param labelImage
     */
    public void setLabelImage(ImagePlus labelImage) {
        this.labelImage = labelImage;
    }

    /**
     * Returns wether a classifier has been trained
     * @return
     */
    public boolean isClassifierTrained() {
        return classifierTrained;
    }

}
