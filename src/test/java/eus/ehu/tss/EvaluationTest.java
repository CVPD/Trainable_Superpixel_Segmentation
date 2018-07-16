package eus.ehu.tss;


import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.SaveDialog;
import ij.plugin.ChannelSplitter;
import ij.plugin.Converter;
import ij.process.ColorSpaceConverter;
import ij.process.ImageConverter;
import ij.process.ImageProcessor;
import ij.process.LUT;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.AggregateableEvaluation;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.awt.*;
import java.io.*;
import java.util.ArrayList;
import java.util.zip.GZIPOutputStream;

public class EvaluationTest {
    public static void main(final String[] args) {
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

        System.out.println("Open input image 1");
        ImagePlus inImage = IJ.openImage();
        System.out.println("Open superpixel image 1");
        ImagePlus spImage = IJ.openImage();
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
        /*int[] rice = new int[4];
        rice[0] = 30; rice[1]=43; rice[2]=96;rice[3]=99;
        int[] background = new int[1];
        background[0] = 1;
        ArrayList<int[]> tags = new ArrayList<>();
        tags.add(background);
        tags.add(rice);
        classes.add("Background");
        classes.add("Rice");*/
        ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList<>();
        String[] selectedFs = RegionFeatures.Feature.getAllLabels();
        for(int i=0;i<selectedFs.length;++i){
            selectedFeatures.add(RegionFeatures.Feature.fromLabel(selectedFs[i]));
        }
        RandomForest classifier = new RandomForest();
        RegionColorFeatures.calculateFeaturesTable(inImage,spImage,selectedFeatures);
    }

    public static Instances merge(Instances data1, Instances data2) throws Exception {
        int asize = data1.numAttributes();
        boolean[] strings_pos = new boolean[asize];

        for(int i = 0; i < asize; ++i) {
            Attribute att = data1.attribute(i);
            strings_pos[i] = att.type() == 2 || att.type() == 1;
        }

        Instances dest = new Instances(data1);
        dest.setRelationName(data1.relationName() + "+" + data2.relationName());
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(data2);
        Instances instances = source.getStructure();
        Instance instance = null;

        while(source.hasMoreElements(instances)) {
            instance = source.nextElement(instances);
            dest.add(instance);

            for(int i = 0; i < asize; ++i) {
                if(strings_pos[i]) {
                    dest.instance(dest.numInstances() - 1).setValue(i, instance.stringValue(i));
                }
            }
        }

        return dest;
    }
}
