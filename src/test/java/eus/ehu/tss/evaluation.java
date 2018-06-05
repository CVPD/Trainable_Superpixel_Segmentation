package eus.ehu.tss;


import ij.IJ;
import ij.ImagePlus;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;

public class evaluation {
    public static void main(final String[] args){

        ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList<>();
        String[] selectedFs = RegionFeatures.Feature.getAllLabels();
        for(int i=0;i<selectedFs.length;++i){
            selectedFeatures.add(RegionFeatures.Feature.fromLabel(selectedFs[i]));
        }

        final ArrayList<String> classes = new ArrayList<String>();
        classes.add("background");
        classes.add("blue");
        classes.add("red");

        for(int i=0;i<10;++i){

            RandomForest exampleClassifier = new RandomForest();
            ImagePlus testImage = null;
            ImagePlus supTest = null;
            ImagePlus gtTest = null;

            System.out.println("Test class: "+(i+1));

            Instances trainingData;
            if(i==0){
                testImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-01.png" ).getFile() );
                supTest = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-01.zip" ).getFile() );
                gtTest = IJ.openImage(TestSuperpixelSegmentation.class.getResource("/eval/groundtruth/groundtruth-01.png").getFile());
                ImagePlus trainingImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-02.png" ).getFile() );
                ImagePlus supImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-02.zip" ).getFile() );
                ImagePlus gtImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/groundtruth/groundtruth-02.png" ).getFile() );
                System.out.println("\tCalculating features for image: 2");
                trainingData = RegionColorFeatures.calculateLabeledColorFeatures(trainingImage1,supImage1,gtImage,selectedFeatures,classes);
                for(int j=3;j<10;++j){
                    trainingImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-0"+j+".png" ).getFile() );
                    supImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-0"+j+".zip" ).getFile() );
                    gtImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/groundtruth/groundtruth-0"+j+".png" ).getFile() );
                    try {
                        System.out.println("\tCalculating features for image: "+j);
                        trainingData = merge(trainingData,RegionColorFeatures.calculateLabeledColorFeatures(trainingImage1,supImage1,gtImage,selectedFeatures,classes));
                    }catch (Exception e){
                        e.printStackTrace();
                    }
                }
                trainingImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-10.png" ).getFile() );
                supImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-10.zip" ).getFile() );
                gtImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/groundtruth/groundtruth-10.png" ).getFile() );
                try {
                    System.out.println("\tCalculating features for image: 10");
                    trainingData = merge(trainingData,RegionColorFeatures.calculateLabeledColorFeatures(trainingImage1,supImage1,gtImage,selectedFeatures,classes));
                }catch (Exception e){
                    e.printStackTrace();
                }
            }else if(i==9) {
                testImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-10.png" ).getFile() );
                supTest = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-10.zip" ).getFile() );
                gtTest = IJ.openImage(TestSuperpixelSegmentation.class.getResource("/eval/groundtruth/groundtruth-10.png").getFile());
                ImagePlus trainingImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-01.png" ).getFile() );
                ImagePlus supImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-01.zip" ).getFile() );
                ImagePlus gtImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/groundtruth/groundtruth-01.png" ).getFile() );
                System.out.println("\tCalculating features for image: 1");
                trainingData = RegionColorFeatures.calculateLabeledColorFeatures(trainingImage1,supImage1,gtImage,selectedFeatures,classes);
                for(int j=2;j<10;++j){
                    trainingImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-0"+j+".png" ).getFile() );
                    supImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-0"+j+".zip" ).getFile() );
                    gtImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/groundtruth/groundtruth-0"+j+".png" ).getFile() );
                    try {
                        System.out.println("\tCalculating features for image: "+j);
                        trainingData = merge(trainingData,RegionColorFeatures.calculateLabeledColorFeatures(trainingImage1,supImage1,gtImage,selectedFeatures,classes));
                    }catch (Exception e){
                        e.printStackTrace();
                    }
                }
            }else {
                testImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-0"+(i+1)+".png" ).getFile() );
                supTest = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-0"+(i+1)+".zip" ).getFile() );
                gtTest = IJ.openImage(TestSuperpixelSegmentation.class.getResource("/eval/groundtruth/groundtruth-0"+(i+1)+".png").getFile());
                ImagePlus trainingImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-01.png" ).getFile() );
                ImagePlus supImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-01.zip" ).getFile() );
                ImagePlus gtImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/groundtruth/groundtruth-01.png" ).getFile() );
                System.out.println("\tCalculating features for image: 1");
                trainingData = RegionColorFeatures.calculateLabeledColorFeatures(trainingImage1,supImage1,gtImage,selectedFeatures,classes);
                for(int j=2;j<10;++j){
                    if(i+1!=j) {
                        trainingImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-0"+j+".png" ).getFile() );
                        supImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-0"+j+".zip" ).getFile() );
                        gtImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/groundtruth/groundtruth-0"+j+".png" ).getFile() );
                        try {
                            System.out.println("\tCalculating features for image: "+j);
                            trainingData = merge(trainingData,RegionColorFeatures.calculateLabeledColorFeatures(trainingImage1,supImage1,gtImage,selectedFeatures,classes));
                        }catch (Exception e){
                            e.printStackTrace();
                        }
                    }
                }
                trainingImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-10.png" ).getFile() );
                supImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-10.zip" ).getFile() );
                gtImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/groundtruth/groundtruth-10.png" ).getFile() );
                try {
                    System.out.println("\tCalculating features for image: 10");
                    trainingData = merge(trainingData,RegionColorFeatures.calculateLabeledColorFeatures(trainingImage1,supImage1,gtImage,selectedFeatures,classes));
                }catch (Exception e){
                    e.printStackTrace();
                }
            }
            System.out.println("\tCalculating features for testing image");
            Instances testingData = RegionColorFeatures.calculateLabeledColorFeatures(testImage,supTest,gtTest,selectedFeatures,classes);
            try {
                System.out.println("\tTraining classifier");
                Evaluation eval = new Evaluation(trainingData);
                exampleClassifier.buildClassifier(trainingData);
                System.out.print("\tEvaluating model");
                eval.evaluateModel(exampleClassifier,testingData);
                System.out.println("Test result: ");
                System.out.println(eval.toSummaryString("\n\t"+(i+1)+" results\n======\n",false));
            }catch (Exception e){
                e.printStackTrace();
            }
            try {
                System.out.println("\tSaving training file");
                BufferedWriter writer = new BufferedWriter(new FileWriter("trainingData"+i+".arff"));
                writer.write(trainingData.toString());
                writer.flush();
                writer.close();
            }catch (Exception e){
                e.printStackTrace();
            }
            try {
                System.out.println("\tSaving testing file");
                BufferedWriter writer = new BufferedWriter(new FileWriter("testingData"+i+".arff"));
                writer.write(testingData.toString());
                writer.flush();
                writer.close();
            }catch (Exception e) {
                e.printStackTrace();
            }
            trainingData=null;
            testingData=null;
            testImage = null;
            supTest = null;
            gtTest = null;
            System.gc();
        }


    }

    public static Instances merge(Instances data1, Instances data2)
            throws Exception
    {
        // Check where are the string attributes
        int asize = data1.numAttributes();
        boolean strings_pos[] = new boolean[asize];
        for(int i=0; i<asize; i++)
        {
            Attribute att = data1.attribute(i);
            strings_pos[i] = ((att.type() == Attribute.STRING) ||
                    (att.type() == Attribute.NOMINAL));
        }

        // Create a new dataset
        Instances dest = new Instances(data1);
        dest.setRelationName(data1.relationName() + "+" + data2.relationName());

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(data2);
        Instances instances = source.getStructure();
        Instance instance = null;
        while (source.hasMoreElements(instances)) {
            instance = source.nextElement(instances);
            dest.add(instance);

            // Copy string attributes
            for(int i=0; i<asize; i++) {
                if(strings_pos[i]) {
                    dest.instance(dest.numInstances()-1)
                            .setValue(i,instance.stringValue(i));
                }
            }
        }

        return dest;
    }
}
