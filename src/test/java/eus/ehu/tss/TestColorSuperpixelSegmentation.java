package eus.ehu.tss;

import ij.IJ;
import ij.ImagePlus;
import ij.io.SaveDialog;
import ij.process.LUT;
import weka.classifiers.trees.RandomForest;
import weka.core.converters.ArffSaver;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;


public class TestColorSuperpixelSegmentation {
    public static void main(final String[] args){
        ImagePlus inputImage = IJ.openImage();
        inputImage.show();
        ImagePlus labelImage = IJ.openImage();
        labelImage.show();

        //LUT
        final byte[] red = new byte[ 256 ];
        final byte[] green = new byte[ 256 ];
        final byte[] blue = new byte[ 256 ];

        // assign colors to classes
        Color[] colors = new Color[ 500 ];

        // hue for assigning new color ([0.0-1.0])
        float hue = 0f;
        // saturation for assigning new color ([0.5-1.0])
        float saturation = 1f;

        // first color is red: HSB( 0, 1, 1 )

        for(int i=0; i<500; i++)
        {
            colors[ i ] = Color.getHSBColor(hue, saturation, 1);

            hue += 0.38197f; // golden angle
            if (hue > 1)
                hue -= 1;
            saturation += 0.38197f; // golden angle
            if (saturation > 1)
                saturation -= 1;
            saturation = 0.5f * saturation + 0.5f;
        }

        for(int i = 0 ; i < 256; i++)
        {
            //IJ.log("i = " + i + " color index = " + colorIndex);
            red[i] = (byte) colors[ i ].getRed();
            green[i] = (byte) colors[ i ].getGreen();
            blue[i] = (byte) colors[ i ].getBlue();
        }
        LUT overlayLUT = new LUT(red, green, blue);


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
        TrainableSuperpixelSegmentation tss = new TrainableSuperpixelSegmentation(inputImage,labelImage,selectedFeatures,exampleClassifier,classes);
        tss.calculateRegionFeatures();
        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(tss.getUnlabeled());
            SaveDialog sd = new SaveDialog("Save instances as...","unlabeled",".arff");
            if(sd.getFileName()==null){
                return;
            }
            String filename = sd.getDirectory() + sd.getFileName();
            saver.setFile(new File(filename));
            saver.writeBatch();
            IJ.log("File saved at "+sd.getDirectory()+sd.getFileName());
        } catch (IOException e) {
            e.printStackTrace();
        }
        tss.calculateTrainingData(tags);
        tss.trainClassifier();
        ImagePlus resultImg = tss.applyClassifier();
        resultImg.show();
        inputImage.show();
        labelImage.getProcessor().setColorModel(overlayLUT);
        labelImage.setDisplayRange(0,tss.getUnlabeled().numInstances());
        labelImage.show();
        ImagePlus rs = tss.getFeatureImage(labelImage,tss.getUnlabeledTable());
        rs.show();


    }
}