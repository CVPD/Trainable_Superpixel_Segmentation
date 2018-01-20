package com.EHU.imagej;

/**
 * This class will hold the features that each superpixel region has.
 * This features will be the intensity measures from the MorphoLibJ library.
 */

import inra.ijpb.measure.IntensityMeasures;

public class RegionFeatures {

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
         * Based on provided label returns Feature with that label
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

}
