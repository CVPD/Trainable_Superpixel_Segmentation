package eus.ehu.tss;


import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Roi;
import ij.measure.ResultsTable;
import ij.process.FloatProcessor;
import ij.process.ImageConverter;
import ij.process.ShortProcessor;
import ij.process.ImageProcessor;
import ij.process.StackConverter;
import inra.ijpb.data.image.Images3D;
import inra.ijpb.label.LabelImages;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Class with utility methods.
 */
public class Utils {

    /**
     * Merge two set of instances
     * @param data1 first set of instances
     * @param data2 second set of instances
     * @return Instances object containing instances of both datasets
     * @throws Exception instances mismatch
     */
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

    /**
     * Calculates coordinates corresponding to labels in label image
     * @param labelImage input image with labels
     * @return a HashMap where the key is the label and the values are the coordinates of the label
     */
    public static HashMap<Integer,int[]> calculateLabelCoordinates(ImagePlus labelImage){
        HashMap<Integer, int[]> result = new HashMap<>();
        final int width = labelImage.getWidth();
        final int height = labelImage.getHeight();

        final int numSlices = labelImage.getImageStackSize();
        for( int z=1; z <= numSlices; z++ )
        {
            final ImageProcessor labelsIP = labelImage.getImageStack().getProcessor( z );

            for( int x = 0; x<width; x++ )
                for( int y = 0; y<height; y++ )
                {
                    int labelValue = (int) labelsIP.getPixelValue( x, y );
                    int[] coord = new int[3];
                    coord[0] = x; coord[1] = y; coord[2] = z;
                    result.putIfAbsent(labelValue,coord);
                }
        }
        return result;
    }

    /**
     * Checks whether a label image contains any pixel with value 0.
     * MorphoLibJ treats 0 as background, but superpixel algorithms such as SLIC
     * usually assign 0 to their first superpixel.
     * @param labelImage input label image
     * @return true if at least one pixel has value 0
     */
    public static boolean hasZeroLabel(ImagePlus labelImage){
        final ImageStack stack = labelImage.getImageStack();
        final int width = stack.getWidth();
        final int height = stack.getHeight();
        for(int z=1; z<=stack.getSize(); z++){
            final ImageProcessor ip = stack.getProcessor(z);
            for(int y=0; y<height; y++)
                for(int x=0; x<width; x++)
                    if(ip.getf(x,y) == 0)
                        return true;
        }
        return false;
    }

    /**
     * Gets all labels of a label image in ascending order, including the 0 label if present.
     * @param labelImage input label image
     * @return sorted array of labels
     */
    public static int[] getAllLabels(ImagePlus labelImage){
        int[] labels = LabelImages.findAllLabels(labelImage);
        if(!hasZeroLabel(labelImage))
            return labels;
        int[] result = new int[labels.length+1];
        result[0] = 0;
        System.arraycopy(labels, 0, result, 1, labels.length);
        return result;
    }

    /**
     * Maps each label to its index in the (sorted) array of labels
     * @param labels array of labels
     * @return map from label to index
     */
    public static HashMap<Integer,Integer> mapLabelIndices(int[] labels){
        HashMap<Integer,Integer> map = new HashMap<>();
        for(int i=0; i<labels.length; i++)
            map.put(labels[i], i);
        return map;
    }

    /**
     * Creates a copy of the label image where all labels are increased by one,
     * so no region is 0 (background for MorphoLibJ).
     * @param labelImage input label image
     * @return new label image with shifted labels
     */
    public static ImagePlus shiftLabels(ImagePlus labelImage){
        final ImageStack stack = labelImage.getImageStack();
        final int width = stack.getWidth();
        final int height = stack.getHeight();
        double max = 0;
        for(int z=1; z<=stack.getSize(); z++){
            final ImageProcessor ip = stack.getProcessor(z);
            for(int y=0; y<height; y++)
                for(int x=0; x<width; x++)
                    max = Math.max(max, ip.getf(x,y));
        }
        final boolean useShort = max + 1 <= 65535;
        final ImageStack result = new ImageStack(width, height);
        for(int z=1; z<=stack.getSize(); z++){
            final ImageProcessor ip = stack.getProcessor(z);
            final ImageProcessor out = useShort ? new ShortProcessor(width, height) : new FloatProcessor(width, height);
            for(int y=0; y<height; y++)
                for(int x=0; x<width; x++)
                    out.setf(x, y, ip.getf(x,y) + 1);
            result.addSlice(stack.getSliceLabel(z), out);
        }
        final ImagePlus shifted = new ImagePlus(labelImage.getTitle(), result);
        shifted.setCalibration(labelImage.getCalibration());
        return shifted;
    }

    /**
     * Applies a look-up table to a label image, taking into account the 0 label
     * (if present in the image it is treated as one more region).
     * @param labelStack label image stack
     * @param values one value per label, ordered as the sorted labels (0 first if present)
     * @return 32-bit stack with the value of each region
     */
    public static ImageStack applyLut(ImageStack labelStack, double[] values){
        final int width = labelStack.getWidth();
        final int height = labelStack.getHeight();
        final int size = labelStack.getSize();
        final ImagePlus tmp = new ImagePlus("labels", labelStack);
        final HashMap<Integer,Integer> indices = mapLabelIndices(getAllLabels(tmp));
        final ImageStack result = ImageStack.create(width, height, size, 32);
        for(int z=0; z<size; z++)
            for(int y=0; y<height; y++)
                for(int x=0; x<width; x++){
                    Integer index = indices.get((int) labelStack.getVoxel(x,y,z));
                    result.setVoxel(x, y, z, (index == null || index >= values.length) ? Double.NaN : values[index]);
                }
        return result;
    }

    /**
     * Gets the labels selected by a ROI, including the 0 label if present in the label image
     * (MorphoLibJ's LabelImages.getSelectedLabels always ignores 0).
     * @param labelImage label image (its ROI is set to the provided one)
     * @param roi selection
     * @return list of selected labels
     */
    public static ArrayList<Float> getSelectedLabels(ImagePlus labelImage, Roi roi){
        if(!hasZeroLabel(labelImage))
            return LabelImages.getSelectedLabels(labelImage, roi);
        // shift labels so 0 is not ignored, then restore the original values
        final ImagePlus shifted = shiftLabels(labelImage);
        shifted.setPosition(labelImage.getCurrentSlice());
        final ArrayList<Float> selected = LabelImages.getSelectedLabels(shifted, roi);
        final ArrayList<Float> result = new ArrayList<>();
        for(Float f : selected)
            result.add(f - 1f);
        labelImage.setRoi(roi);
        return result;
    }

    /**
     * Merge two Results Tables assuming both have the same columns
     * @param rs1 first results table to merge
     * @param rs2 second results table to merge
     * @return resulting results table
     */
    public static ResultsTable mergeResultsTables(ResultsTable rs1, ResultsTable rs2){
        ResultsTable result = (ResultsTable) rs1.clone();
        for(int i=0;i<rs2.getCounter();++i){
            result.incrementCounter();
            result.addLabel(rs2.getLabel(i));
            for(int j=0;j<rs2.getLastColumn();++j){
                result.addValue(j,rs2.getValueAsDouble(j,i));
            }
        }
        return result;
    }

    /**
     * Remaps label image so that regions in different slices have different values
     * @param labelImage label image
     * @return label image with different regions each slice
     */
    public static ImagePlus remapLabelImage(ImagePlus labelImage){
        ImageStack img = labelImage.getStack();
        double max = 0;
        double prevMax = 0;
        for(int z=0;z<img.getSize();++z){
            ImageProcessor slice = img.getProcessor(z+1);
            LabelImages.remapLabels(slice);
            for(int y = 0; y < slice.getHeight();++y){
                for(int x = 0;x<slice.getWidth();++x){
                    double p = slice.getf(x,y);
                    if(p!=0) {
                        if (p > max) {
                            max = p;
                        }
                        img.setVoxel(x, y, z, p + prevMax);
                    }
                }
            }
            prevMax+=max;
            max=0;
        }
        ImagePlus result = new ImagePlus(labelImage.getShortTitle(),img);
        Images3D.optimizeDisplayRange(result);
        return result;
    }

    /**
     * Convert image to 8 bit in place without scaling it. (Taken from Weka_Segmentation.)
     *
     * @param image input image
     */
    public static void convertTo8bitNoScaling( ImagePlus image )
    {
        boolean aux = ImageConverter.getDoScaling();

        ImageConverter.setDoScaling( false );

        if( image.getImageStackSize() > 1)
            (new StackConverter( image )).convertToGray8();
        else
            (new ImageConverter( image )).convertToGray8();

        ImageConverter.setDoScaling( aux );
    }

}
