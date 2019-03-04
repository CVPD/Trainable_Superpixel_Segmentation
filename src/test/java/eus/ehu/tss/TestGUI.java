package eus.ehu.tss;


import ij.ImageJ;

/**
 * This class allows testing the plugin GUI.
 *
 */
public class TestGUI {
    /**
     * Main class to test GUI.
     * @param args ImageJ arguments
     */
    public static void main( final String[] args )
    {
        ImageJ.main( args );
        new Trainable_Superpixel_Segmentation().run(null);
    }
}
