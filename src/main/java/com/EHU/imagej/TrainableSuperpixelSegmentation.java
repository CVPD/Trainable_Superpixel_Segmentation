package com.EHU.imagej;

import ij.ImagePlus;

import java.util.ArrayList;

/**
 * Main class of the library that will conduct the classification of the images.
 */
public class TrainableSuperpixelSegmentation {

    private ArrayList<RegionFeatures.Feature> features = new ArrayList<RegionFeatures.Feature>();
    private ImagePlus image;

    /**
     *
     * @param originalImage Original image ImagePlus
     * @param labelImage clustered image ImagePlus
     * @param tags ArrayList of Integers with the tags of the labeled clusters, it's length has to be equal to the number of clusters in labelImage
     */
    public TrainableSuperpixelSegmentation(ImagePlus originalImage, ImagePlus labelImage, ArrayList<Integer> tags){

    }

    /**
     * This method will calculate the selected features for each region
     * @param selectedFeatures
     */
    private void calculateRegionFeatures(ArrayList<RegionFeatures.Feature> selectedFeatures){

    }


}
