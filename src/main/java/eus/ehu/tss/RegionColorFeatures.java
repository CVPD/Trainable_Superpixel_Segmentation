package eus.ehu.tss;

import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.ResultsTable;
import ij.plugin.ChannelSplitter;
import ij.process.ColorSpaceConverter;
import ij.process.ImageProcessor;
import inra.ijpb.measure.ResultsBuilder;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * This class will extract region features from colored images using the Lab color space
 */
public class RegionColorFeatures {

    /**
     * Callable function to calculate unlabeled Instances
     * @param inputImage input image
     * @param labelImage corresponding label image
     * @param selectedFeatures ArrayList of selected features
     * @param classes ArrayList of possible classes
     * @return Callable with calculated instances
     */
    private static Callable<Instances> getUnlabeledInstances(ImagePlus inputImage,
                                                             ImagePlus labelImage,
                                                             ArrayList<RegionFeatures.Feature> selectedFeatures,
                                                             ArrayList<String> classes){
        if(Thread.currentThread().isInterrupted()){
            return null;
        }
        return new Callable<Instances>() {
            @Override
            public Instances call() throws Exception {
                Instances result =  RegionFeatures.calculateUnlabeledRegionFeatures(inputImage,labelImage,selectedFeatures,classes);
                return result;
            }
        };
    }

    /**
     * Callable function to calculate table with region features
     * @param inputImage input image
     * @param labelImage corresponding label image
     * @param selectedFeatures ArrayList of selected features
     * @return Callable with features in ResultsTable format
     */
    private static Callable<ResultsTable> getUnlabeledTables(ImagePlus inputImage,
                                                             ImagePlus labelImage,
                                                             ArrayList<RegionFeatures.Feature> selectedFeatures){
        if(Thread.currentThread().isInterrupted()){
            return null;
        }
        return new Callable<ResultsTable>() {
            @Override
            public ResultsTable call() throws Exception {
                ResultsTable result =  RegionFeatures.calculateFeaturesTable(inputImage,labelImage,selectedFeatures);
                return result;
            }
        };
    }

    /**
     * Callable function to calculate labeled instances
     * @param inputImage input image
     * @param labelImage corresponding label image
     * @param groundtruth groundtruth image
     * @param selectedFeatures ArrayList of selected features
     * @param classes ArrayList of possible classes
     * @return Callable with Instances
     */
    private static Callable<Instances> getLabeledInstances(ImagePlus inputImage,
                                                             ImagePlus labelImage,
                                                             ImagePlus groundtruth,
                                                             ArrayList<RegionFeatures.Feature> selectedFeatures,
                                                             ArrayList<String> classes){
        if(Thread.currentThread().isInterrupted()){
            return null;
        }
        return new Callable<Instances>() {
            @Override
            public Instances call() throws Exception {
                Instances result =  RegionFeatures.calculateLabeledRegionFeatures(inputImage,labelImage,groundtruth,selectedFeatures,classes);
                return result;
            }
        };
    }

    /**
     * Calculates ResultsTable of features for each region in label image
     * @param inputImage input image
     * @param labelImage label image
     * @param selectedFeatures ArrayList with selected features
     * @return ResultsTable with features per region
     */
    public static ResultsTable calculateFeaturesTable(
            ImagePlus inputImage,
            ImagePlus labelImage,
            ArrayList<RegionFeatures.Feature> selectedFeatures) {

        ArrayList<ResultsTable> resultsTables = new ArrayList<>();
        ImageStack lStack = new ImageStack(inputImage.getWidth(),inputImage.getHeight());
        ImageStack aStack = new ImageStack(inputImage.getWidth(),inputImage.getHeight());
        ImageStack bStack = new ImageStack(inputImage.getWidth(),inputImage.getHeight());
        ImageStack stack = inputImage.getStack();
        for(int z=0;z<inputImage.getNSlices();++z){
            ImageProcessor sliceProcessor = stack.getProcessor(z+1);
            ImagePlus slice = new ImagePlus("Slice" + z, sliceProcessor);
            ColorSpaceConverter converter = new ColorSpaceConverter();
            ImagePlus lab = converter.RGBToLab(slice);
            ImagePlus[] channels = ChannelSplitter.split(lab);
            lStack.addSlice(channels[0].getProcessor());
            aStack.addSlice(channels[1].getProcessor());
            bStack.addSlice(channels[2].getProcessor());
        }
        ExecutorService exe = Executors.newFixedThreadPool(3);
        final ArrayList<Future<ResultsTable>> futures = new ArrayList<Future<ResultsTable>>();
        try {
            futures.add(exe.submit(getUnlabeledTables(new ImagePlus(inputImage.getShortTitle()+"-l",lStack), labelImage, selectedFeatures)));
            futures.add(exe.submit(getUnlabeledTables(new ImagePlus(inputImage.getShortTitle()+"-a",aStack), labelImage, selectedFeatures)));
            futures.add(exe.submit(getUnlabeledTables(new ImagePlus(inputImage.getShortTitle()+"-b",bStack), labelImage, selectedFeatures)));
            int i = 0;
            for (Future<ResultsTable> f : futures) {
                ResultsTable res = f.get();
                resultsTables.add(res);
            }
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        } finally {
            exe.shutdown();
        }
        for(int i=0;i<resultsTables.get(0).getLastColumn()+1;++i){
            //Rename columns based on channel they belong to
            resultsTables.get(0).renameColumn(resultsTables.get(0).getColumnHeading(i),resultsTables.get(0).getColumnHeading(i)+"-l");

            resultsTables.get(1).renameColumn(resultsTables.get(1).getColumnHeading(i),resultsTables.get(1).getColumnHeading(i)+"-a");

            resultsTables.get(2).renameColumn(resultsTables.get(2).getColumnHeading(i),resultsTables.get(2).getColumnHeading(i)+"-b");
        }
        ResultsBuilder rb = new ResultsBuilder(resultsTables.get(0));
        rb.addResult(resultsTables.get(1));
        rb.addResult(resultsTables.get(2));
        return rb.getResultsTable();
    }

    /**
     * Creates Instances based on resultsTable
     * @param resultsTable table with features
     * @param classes ArrayList with possible classes
     * @return resulting Instances
     */
    public static Instances calculateUnabeledInstances(ResultsTable resultsTable, ArrayList<String> classes){
        Instances unlabeled = RegionFeatures.calculateUnabeledInstances(resultsTable,classes);
        return unlabeled;
    }

    /**
     * Creates Instances object based on the features of each color channel after converting the image to Lab
     * @param inputImage RGB input image
     * @param labelImage Label image
     * @param selectedFeatures ArrayList with selected features from RegionFeatures.Feature
     * @param classes ArrayList of Strings with names of the classes
     * @return Dataset with the features of each color for each region from the labelImage
     */
    public static Instances calculateUnlabeledColorFeatures(ImagePlus inputImage,
                                                            ImagePlus labelImage,
                                                            ArrayList<RegionFeatures.Feature> selectedFeatures,
                                                            ArrayList<String> classes)
    {
        ResultsTable resultsTable = calculateFeaturesTable(inputImage,labelImage,selectedFeatures);
        Instances unlabeled = calculateUnabeledInstances(resultsTable,classes);
        return unlabeled;
    }

    /**
     * Creates Instances object based on the features of each color channel after converting the image to Lab and sets class based on provided Ground Truth image
     * @param inputImage RGB input image
     * @param labelImage Label image
     * @param gtImage Ground Truth image
     * @param selectedFeatures ArrayList with selected features from RegionFeatures.Feature
     * @param classes ArrayList of Strings with names of the classes
     * @return Dataset with the features of each color for each region from the labelImage
     */
    public static Instances calculateLabeledColorFeatures(ImagePlus inputImage,
                                                          ImagePlus labelImage,
                                                          ImagePlus gtImage,
                                                          ArrayList<RegionFeatures.Feature> selectedFeatures,
                                                          ArrayList<String> classes) {
        HashMap<Integer, int[]> labelCoord = Utils.calculateLabelCoordinates(labelImage);
        ResultsTable mergedTable = calculateFeaturesTable(inputImage,labelImage,selectedFeatures);
        //mergedTable.show( inputImage.getShortTitle() + "-intensity-measurements" );
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        int numFeatures = mergedTable.getLastColumn()+1; //Take into account it starts in index 0
        for(int i=0;i<numFeatures;++i){
            attributes.add(new Attribute(mergedTable.getColumnHeading(i),i));
        }
        attributes.add(new Attribute("Class", classes));
        Instances labeled = new Instances("training data",attributes,0);
        for(int i=0;i<mergedTable.size();++i){
            //numFeatures is the index, add 1 to get number of attributes needed plus class
            Instance inst = new DenseInstance(numFeatures+1);
            for(int j=0;j<numFeatures;++j){
                inst.setValue(j,mergedTable.getValueAsDouble(j,i));
            }
            int[] coord = labelCoord.get(i+1);
            ImageProcessor gtProcessor = gtImage.getProcessor();
            float value = (float) gtProcessor.getf(coord[0],coord[1]);
            inst.setValue( numFeatures, (int) value );
            labeled.add(inst);
        }
        labeled.setClassIndex( numFeatures );
        //The number or instances should be the same as the size of the table
        if(labeled.numInstances()!=(mergedTable.size())){
            return null;
        }else{
            return labeled;
        }
    }
}
