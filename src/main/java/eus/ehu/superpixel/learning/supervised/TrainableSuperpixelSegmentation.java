package eus.ehu.superpixel.learning.supervised;

import ij.ImagePlus;
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


    /**
     * Creates instance of TrainableSuperpixelSegmentation based on an image and it's corresponding label image, a list of selected features and an AbstractClassifier
     * @param originalImage ImagePlus image that will be analyzed
     * @param labels ImagePlus labeled image of the originalImage
     * @param features ArrayList of Features (from RegionFeatures.Feature) that represent the features that will be calculated
     * @param classifier AbstractClassifier that will be used to classify the images
     */
    public TrainableSuperpixelSegmentation(ImagePlus originalImage, ImagePlus labels, ArrayList<RegionFeatures.Feature> features, AbstractClassifier classifier){
        selectedFeatures = features;
        inputImage = originalImage;
        labelImage = labels;
        abstractClassifier  = classifier;
        if(!this.calculateRegionFeatures()){
            System.out.println("Error when calculating Region Features");
        }
    }

    /**
     * Calculates the selected features for each region and saves them on the private variable unlabeled
     */
    private boolean calculateRegionFeatures(){
        unlabeled = RegionFeatures.calculateRegionFeatures(inputImage,labelImage,selectedFeatures);
        return unlabeled != null;
    }

    /**
     * Outputs the features by region through a results table and through printing the created Instances
     */
    public void showFeaturesByRegion(){
        System.out.println(unlabeled.toString());
    }

    /**
     * Trains classifiers based on previously created features and a list of classes with their corresponding regions
     * @param classRegions ArrayList of int[] where each int[] represents the labels of superpixels that belong to the class indicated by their index in the ArrayList
     */
    public boolean trainClassifier(ArrayList<int[]> classRegions){

        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        int numFeatures = unlabeled.numAttributes()-1;
        for(int i=0;i<numFeatures;++i){
            attributes.add(new Attribute(unlabeled.attribute(i).name(),i));
        }
        attributes.add(new Attribute("Class"));
        Instances trainingData = new Instances("training data",attributes,0);
        for(int i=0;i<classRegions.size();++i){ //For each class in classRegions
            for(int j=0;j<classRegions.get(i).length;++j){
                Instance inst = new DenseInstance(numFeatures+1);
                for(int k=0;k<(numFeatures);++k){
                    inst.setValue(k,unlabeled.get(classRegions.get(i)[j]).value(k));
                }
                inst.setValue(numFeatures,i);
                trainingData.add(inst);
            }
        }
        trainingData.setClassIndex(numFeatures); //Index inside the attribute array for class is equal to number of features
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
     * Applies classifier to unlabeled data and creates ne Instances in labeled private variables
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
        ImageProcessor ip = labelImage.getProcessor();
        for (int x=0;x<width;++x){
            for(int y=0;y<height;++y){
                int index = ip.getPixel(x,y);
                if(index==0){ //edge pixel
                    tags[x+y*width]= index;
                }else {
                    Instance instance = labeled.get(index - 1);
                    tags[x+y*width]= (float) instance.classValue();
                }
            }
        }
        FloatProcessor processor = new FloatProcessor(width,height,tags);
        ImagePlus result = new ImagePlus("Labeled image",processor);
        return result;
    }


}
