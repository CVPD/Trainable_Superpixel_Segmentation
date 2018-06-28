package eus.ehu.tss;

import ij.IJ;
import ij.ImagePlus;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.util.ArrayList;


public class TestColorSuperpixelSegmentation {
    public static void main(final String[] args){
        ImagePlus inputImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/TMA3d.tif" ).getFile() );
        inputImage.show();
        ImagePlus labelImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/TMA-Segmentation-16-bit3d.tif").getFile() );
        labelImage.show();


        // Use all features
        ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList<>();
        String[] selectedFs = RegionFeatures.Feature.getAllLabels();
        for(int i=0;i<selectedFs.length;++i){
            selectedFeatures.add(RegionFeatures.Feature.fromLabel(selectedFs[i]));
        }

        // Define 3 classes ("No-Stained-No-Tumor", "Stained-Tumor" and "Background")
        final ArrayList<String> classes = new ArrayList<String>();
        classes.add("Background");
        classes.add( "No-Stained-No-Tumor" );
        classes.add( "Stained-Tumor" );

        // Define classifier
        RandomForest exampleClassifier = new RandomForest();
        TrainableSuperpixelSegmentation tss  = new TrainableSuperpixelSegmentation(inputImage,labelImage,selectedFeatures,exampleClassifier,classes);
        tss.calculateRegionFeatures();
        ImagePlus result = tss.applyClassifier();


    }
}