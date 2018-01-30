package com.EHU.imagej;

import ij.ImagePlus;
import ij.measure.ResultsTable;
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
     *
     * @param originalImage Original image ImagePlus
     * @param labelImage clustered image ImagePlus
     * @param tags ArrayList of Integers with the tags of the labeled clusters, it's length has to be equal to the number of clusters in labelImage
     */
    public TrainableSuperpixelSegmentation(ImagePlus originalImage, ImagePlus labels, ArrayList<Integer> tags, ArrayList<RegionFeatures.Feature> features){
        selectedFeatures = features;
        inputImage = originalImage;
        labelImage = labels;
        this.calculateRegionFeatures();
    }

    /**
     * This method will calculate the selected features for each region
     * @param selectedFeatures
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

    public void showFeaturesByRegion(){
        mergedTable.show( inputImage.getShortTitle() + "-intensity-measurements" );
        System.out.println(instances.toString());
    }


}
