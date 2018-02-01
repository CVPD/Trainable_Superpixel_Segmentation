package com.EHU.imagej;

import ij.ImagePlus;
import ij.measure.ResultsTable;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import inra.ijpb.label.LabelImages;
import org.w3c.dom.Attr;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

/**
 * Main class of the library that will conduct the classification of the images.
 */
public class TrainableSuperpixelSegmentation {

    private ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList<RegionFeatures.Feature>();
    private ImagePlus inputImage;
    private ImagePlus labelImage;
    private ResultsTable mergedTable;
    private Instances unlabeled;
    private Instances labeled;
    private AbstractClassifier trainedClassifier;


    /**
     * Creates instance of TrainableSuperpixelSegmentation based on an image and it's corresponding label image and a list of selected features
     * @param originalImage
     * @param labels
     * @param features
     */
    public TrainableSuperpixelSegmentation(ImagePlus originalImage, ImagePlus labels, ArrayList<RegionFeatures.Feature> features){
        selectedFeatures = features;
        inputImage = originalImage;
        labelImage = labels;
        this.calculateRegionFeatures();
    }

    /**
     * Calculates the selected features for each region and saves them on the private variable unlabeled
     */
    private void calculateRegionFeatures(){
        RegionFeatures features = new RegionFeatures(inputImage,labelImage,selectedFeatures);
        ArrayList<ResultsTable> results = new ArrayList<ResultsTable>();
        for (RegionFeatures.Feature selectedFeature : selectedFeatures) {
            switch (selectedFeature) {
                case Max:
                    results.add( features.getMaxTable() );
                    break;
                case Min:
                    results.add( features.getMinTable() );
                    break;
                case Mean:
                    results.add( features.getMeanTable() );
                    break;
                case Mode:
                    results.add( features.getModeTable() );
                    break;
                case Median:
                    results.add( features.getMedianTable() );
                    break;
                case StdDev:
                    results.add( features.getStdDevTable() );
                    break;
                case Kurtosis:
                    results.add( features.getKurtosisTable() );
                    break;
                case Skewness:
                    results.add( features.getSkewnessTable() );
                    break;
            }
        }
        mergedTable = new ResultsTable();
        final int numLabels = results.get( 0 ).getCounter();
        for(int i=0; i < numLabels; ++i)
        {
            mergedTable.incrementCounter();
            String label = results.get( 0 ).getLabel( i );
            mergedTable.addLabel(label);

            for (ResultsTable result : results) {
                String measure = result.getColumnHeading(0);
                double value = result.getValue(measure, i);
                mergedTable.addValue(measure, value);
            }
        }
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        int numFeatures = mergedTable.getLastColumn(); //Take into account it starts in index 0
        for(int i=0;i<numFeatures+1;++i){
            attributes.add(new Attribute(mergedTable.getColumnHeading(i),i));
        }
        attributes.add(new Attribute("Class"));
        unlabeled = new Instances("dataset",attributes,0);
        for(int i=0;i<mergedTable.size();++i){
            Instance inst = new DenseInstance(numFeatures+2);//numFeatures is the index, add 1 to get number of attributes needed plus class
            for(int j=0;j<(numFeatures+1);++j){
                inst.setValue(j,mergedTable.getValue(j,i));
            }
            inst.setValue(numFeatures+1,0);//set class as 0
            unlabeled.add(inst);
        }
        unlabeled.setClassIndex(numFeatures+1);
    }

    /**
     * Outputs the features by region through a results table and through printing the created Instances
     */
    public void showFeaturesByRegion(){
        mergedTable.show( inputImage.getShortTitle() + "-intensity-measurements" );
        System.out.println(unlabeled.toString());
    }

    /**
     * Trains classifiers based on previously created features and a list of classes with their corresponding regions
     * @param classifier
     * @param classRegions
     */
    public void trainClassifier(AbstractClassifier classifier, ArrayList<int[]> classRegions){

        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        int numFeatures = mergedTable.getLastColumn(); //Take into account it starts in index 0
        for(int i=0;i<numFeatures+1;++i){
            attributes.add(new Attribute(mergedTable.getColumnHeading(i),i));
        }
        attributes.add(new Attribute("Class"));
        Instances trainingData = new Instances("training data",attributes,0);
        for(int i=0;i<classRegions.size();++i){ //For each class in classRegions
            for(int j=0;j<classRegions.get(i).length;++j){
                Instance inst = new DenseInstance(numFeatures+2);//numFeatures is the index, add 2 to get number of attributes needed plus class
                for(int k=0;k<(numFeatures+1);++k){
                    inst.setValue(k,mergedTable.getValue(k,classRegions.get(i)[j]));
                }
                inst.setValue(numFeatures+1,i);
                trainingData.add(inst);
            }
        }
        trainingData.setClassIndex(numFeatures+1);
        try {
            classifier.buildClassifier(trainingData);
        } catch (Exception e) {
            System.out.println("Error when building classifier");
            e.printStackTrace();
        }
        trainedClassifier = classifier;
    }

    /**
     * Applies classifier to unlabeled data and creates ne Instances in labeled private variables
     */
    public void applyClassifier(){
        try {
            labeled = new Instances(unlabeled); //Copy of unlabeled to label
            for (int i = 0; i < unlabeled.numInstances(); ++i) {
                double classLabel = trainedClassifier.classifyInstance(unlabeled.instance(i));
                labeled.instance(i).setClassValue(classLabel);
            }
            System.out.println(labeled.toString());
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
        result.show();
    }


}
