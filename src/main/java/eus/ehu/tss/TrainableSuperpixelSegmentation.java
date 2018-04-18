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
    private Instances unlabeled;
    private Instances labeled;
    private AbstractClassifier abstractClassifier;
    ArrayList<String> classes = null;


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
        if(!this.calculateRegionFeatures()){
            System.out.println("Error when calculating Region Features");
        }
    }

    /**
     * Calculates the selected features for each region and saves them on the private variable unlabeled
     * @return boolean that checks if the region features have been created
     */
    private boolean calculateRegionFeatures(){
        if(inputImage.getType()==ImagePlus.COLOR_RGB){
            unlabeled = RegionColorFeatures.calculateColorFeatures(
                    inputImage,
                    labelImage,
                    selectedFeatures,
                    classes);
        }else {
            unlabeled = RegionFeatures.calculateRegionFeatures(
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
        Instances trainingData = new Instances("training data",attributes,0);
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
        return new ImagePlus(inputImage.getShortTitle()+"-supseg",result);
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

}
