package eus.ehu.tss;

import ij.IJ;
import ij.ImagePlus;
import weka.classifiers.trees.RandomForest;

import java.util.ArrayList;

public class TestSuperpixelSegmentation{
    public static void main(final String[] args){
        ImagePlus inputImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/TMA-8bit.tif" ).getFile() );
        inputImage.show();
        ImagePlus labelImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/Segmentation3d.tif").getFile() );
        labelImage.show();
        
        // Use all features
        ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList<>();
        String[] selectedFs = RegionFeatures.Feature.getAllLabels();
        for(int i=0;i<selectedFs.length;++i){
            selectedFeatures.add(RegionFeatures.Feature.fromLabel(selectedFs[i]));
        }


        //System.out.println(test.getFeaturesByRegion());

        // Define training regions (one for background and 4 for rice grains)
        final ArrayList<String> classes = new ArrayList<String>();
        classes.add("Background");
        classes.add( "No-Stained-No-Tumor" );
        classes.add( "Stained-Tumor" );
        int[] background = new int[9];
        background[0] = 1; background[1] = 2; background[2]=5;background[3]=6;background[4]=7;background[5]=4;background[6]=89;background[7]=185;background[8]=93;
        int[] tumoral = new int[5];
        tumoral[0] = 13; tumoral[1] = 14; tumoral[2] = 25; tumoral[3] = 32; tumoral[4]=175;
        int[] nontumoral = new int[4];
        nontumoral[0] = 15; nontumoral[1] = 20; nontumoral[2] = 16; nontumoral[3] = 47;
        // Define training regions (one for background and 4 for rice grains)
        ArrayList<int[]> tags = new ArrayList<>();
        tags.add(background);
        tags.add(tumoral);
        tags.add(nontumoral);


        // Define classifier
        RandomForest exampleClassifier = new RandomForest();
        TrainableSuperpixelSegmentation test =
                new TrainableSuperpixelSegmentation( inputImage, labelImage,
                        selectedFeatures, exampleClassifier, classes );

        //test.calculateRegionFeatures();
        // Train classifier using those labels
        if(test.calculateTrainingData(tags)){
            test.trainClassifier();
            ImagePlus result = test.applyClassifier();
            result.show();
        }
        ImagePlus rs = test.getFeatureImage(labelImage,test.getUnlabeledTable());
        rs.show();
    }
}