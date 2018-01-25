package com.EHU.imagej;


import ij.ImagePlus;
import ij.measure.ResultsTable;
import inra.ijpb.measure.IntensityMeasures;

import java.util.ArrayList;


/**
 * This class will hold the features that each superpixel region has.
 * This features will be the intensity measures from the MorphoLibJ library.
 */
public class RegionFeatures {

    private ResultsTable maxTable = null;
    private ResultsTable minTable = null;
    private ResultsTable meanTable = null;
    private ResultsTable modeTable = null;
    private ResultsTable medianTable = null;
    private ResultsTable stdDevTable = null;
    private ResultsTable kurtosisTable = null;
    private ResultsTable skewnessTable = null;

    public ResultsTable getMaxTable() {
        return maxTable;
    }

    public ResultsTable getMinTable() {
        return minTable;
    }

    public ResultsTable getMeanTable() {
        return meanTable;
    }

    public ResultsTable getModeTable() {
        return modeTable;
    }

    public ResultsTable getMedianTable() {
        return medianTable;
    }

    public ResultsTable getStdDevTable() {
        return stdDevTable;
    }

    public ResultsTable getKurtosisTable() {
        return kurtosisTable;
    }

    public ResultsTable getSkewnessTable() {
        return skewnessTable;
    }


    /**
     * enum that lists the Features that can be obtained from the MorphoLibJ Intensity Measures
     */
    public enum Feature{
        Mean("Mean"),
        Median("Median"),
        Mode("Mode"),
        Skewness("Skewness"),
        Kurtosis("Kurtosis"),
        StdDev("StdDev"),
        Max("Max"),
        Min("Min");

        private final String label;

        /**
         * Create feature with a label
         * @param label String with the label of the feature to be created
         */
        private Feature(String label) {this.label = label;}

        /**
         * Returns string with the value of the label
         * @return String with the value of the label
         */
        public String toString(){ return this.label;}

        /**
         * Gets a String array with all labels that are listed in this enum
         * @return String array with values of all labels
         */
        public static String[] getAllLabels(){
            int n = Feature.values().length;
            String[] result = new String[n];
            int i=0;
            for(Feature feature : Feature.values()){
                result[i++] = feature.label;
            }
            return result;
        }

        /**
         * Gets the number of features that are in this enum
         * @return int with number of features that are in this enum
         */
        public static int numFeatures(){
            int n=0;
            for(String item : getAllLabels()){
                n++;
            }
            return n;
        }

        /**
         * Based on provided label returns Feature with that labelx
         * @param fLabel String with the name of the feature
         * @return Feature which matches the provided String
         */
        public static Feature fromLabel(String fLabel){
            if(fLabel != null){
                fLabel = fLabel.toLowerCase();
                for(Feature feature : Feature.values()){
                    String cmp = feature.label.toLowerCase();
                    if(cmp.equals(fLabel)){
                        return feature;
                    }
                }
                throw new IllegalArgumentException("Unable to parse Feature with label " + fLabel);
            }
            return null;
        }

    };

    /**
     * Calculate the features of every region
     * @param inputImage ImagePlus with the features to be calculated
     * @param labelImage ImagePlus with the regions
     * @param selectedFeatures ArrayList with the features that are going to be calculated
     */
    public RegionFeatures(ImagePlus inputImage, ImagePlus labelImage, ArrayList<Feature> selectedFeatures){

        IntensityMeasures calculator = new IntensityMeasures(inputImage,labelImage);

        /*
        Calculate features for selected features
         */
        for (Feature selectedFeature : selectedFeatures) {
            switch (selectedFeature) {
                case Max:
                    maxTable = calculator.getMax();
                    break;
                case Min:
                    minTable = calculator.getMin();
                    break;
                case Mean:
                    meanTable = calculator.getMean();
                    break;
                case Mode:
                    modeTable = calculator.getMode();
                    break;
                case Median:
                    medianTable = calculator.getMedian();
                    break;
                case StdDev:
                    stdDevTable = calculator.getStdDev();
                    break;
                case Kurtosis:
                    kurtosisTable = calculator.getKurtosis();
                    break;
                case Skewness:
                    skewnessTable = calculator.getSkewness();
                    break;
            }
        }

    }


}
