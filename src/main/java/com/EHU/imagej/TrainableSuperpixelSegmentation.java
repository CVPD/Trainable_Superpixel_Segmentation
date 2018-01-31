package com.EHU.imagej;

import ij.ImagePlus;
import ij.measure.ResultsTable;
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
    private ResultsTable mergedTable;
    private Instances instances;

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
     * Calculates the selected features for each region and saves them on the private variable instances
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

        instances = new Instances("dataset",attributes,0);
        for(int i=0;i<mergedTable.size();++i){
            Instance inst = new DenseInstance(numFeatures+1);//numFeatures is the index, add 1 to get number of attributes needed
            for(int j=0;j<(numFeatures+1);++j){
                inst.setValue(j,mergedTable.getValue(j,i));
            }
            instances.add(inst);
        }
    }

    /**
     * Outputs the features by region through a results table and through printing the created Instances
     */
    public void showFeaturesByRegion(){
        mergedTable.show( inputImage.getShortTitle() + "-intensity-measurements" );
        System.out.println(instances.toString());
    }

    /**
     * Trains classifiers based on previously created features and a list of classes with their corresponding regions
     * @param classifier
     * @param classRegions
     */
    public void trainClassifier(AbstractClassifier classifier, ArrayList<Integer[]> classRegions){



    }


}
