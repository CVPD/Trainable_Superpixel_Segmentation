package com.EHU.imagej;

import ij.ImagePlus;
import ij.measure.ResultsTable;
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
    private ArrayList<Instances> instances;

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

    }

    public void showFeaturesByRegion(){
        mergedTable.show( inputImage.getShortTitle() + "-intensity-measurements" );
    }


}
