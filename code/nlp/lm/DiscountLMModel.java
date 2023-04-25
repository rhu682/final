/*
 * Rohan Huang
 * CS159
 * Assignment 2
 */

package nlp.lm;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner; 
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays; // used to print arrays, may be unused
import java.util.HashSet; 

public class DiscountLMModel implements LMModel {

    ////////////////////////
    /* INSTANCE VARIABLES */
    ////////////////////////
    private HashMap<String, Double> unigrams = new HashMap<String, Double>();
    private HashMap<String, HashMap<String, Double>> bigrams = new HashMap<String, HashMap<String, Double>>();
    private HashMap<String, Double> alphas = new HashMap<String, Double>();
    private double discount;

    ////////////////////
    /* PUBLIC METHODS */
    ////////////////////
    public DiscountLMModel(String filename, double discount){        
        this.discount = discount;

        try {
            File textfile = new File(filename);
            Scanner reader = new Scanner(textfile);
            HashSet<String> vocab = new HashSet<String>();
            int wordCount = 0; // used for unigram

            // adds values to vocab that are guaranteed to show up
            unigrams.put("<UNK>", 0.0);
            unigrams.put("<s>", 0.0);
            unigrams.put("</s>", 0.0);
            bigrams.put("<UNK>", new HashMap<String, Double>());
            bigrams.put("<s>", new HashMap<String, Double>());

            while (reader.hasNextLine()) { // processes one line at a time
                // Adds start and end symbols to the line, and then splits it into an array
                String[] currLine = String.join(" ", "<s>", reader.nextLine(), "</s>").split(" ");

                wordCount += currLine.length;
                // replace new words words with <UNK>
                for (int i = 1; i < currLine.length - 1; i++) {
                    if (!vocab.contains(currLine[i])) {
                        vocab.add(currLine[i]);
                        currLine[i] = "<UNK>";
                    }                
                  }

                
                for (int i = 0; i < currLine.length; i++) {
                    // counts unigrams
                    if (unigrams.containsKey(currLine[i]))
                        unigrams.put(currLine[i], unigrams.get(currLine[i]) + 1);
                    else
                        unigrams.put(currLine[i], 1.0);
                    // count bigrams
                    if (i < currLine.length - 1)
                        bigramAdd(currLine[i], currLine[i+1]);        
                }
            }
            reader.close();
            
            // transform unigram counts into probabilities
            for (String key : unigrams.keySet()) {
                unigrams.put(key, unigrams.get(key) / wordCount);
            }

            // transform bigram counts into probabilties
            for (String first : bigrams.keySet()) {
                // calculate reserved mass
                double reserved_mass = bigrams.get(first).size() * discount;
                double firstXTotal = 0.0;
                for (Double doub : bigrams.get(first).values()) {
                    firstXTotal += doub;
                }
                reserved_mass /= firstXTotal;

                // calculate sum of backed off probability
                double backSum = 0;
                for (String second : bigrams.get(first).keySet()) {
                    backSum += unigrams.get(second);
                }
                backSum = 1 - backSum;

                // calculate alpha
                double alpha = reserved_mass / backSum;
                alphas.put(first, alpha);

                // discount all counts
                for (String second : bigrams.get(first).keySet()) {
                    double count = bigrams.get(first).get(second);
                    count -= discount;
                    bigrams.get(first).put(second, count / firstXTotal);
                }
            }

        // handle any file not found exceptions
        } catch (FileNotFoundException e) {
            System.out.println("File not found.");
            e.printStackTrace();
        }
    }
    
    public double logProb(ArrayList<String> sentWords) {
        ArrayList<String> processedSentence = formatSentence(sentWords);
        double logSum = 0;
        for (int i = 0; i < processedSentence.size() - 1; ++i) {
            String word1 = processedSentence.get(i);
            String word2 = processedSentence.get(i + 1);

            logSum += Math.log10(getBigramProb(word1, word2));
        }
        return logSum;
    }

    public double getPerplexity(String filename) {
        try {
            File textfile = new File(filename);
            Scanner reader = new Scanner(textfile);
            double logSum = 0;
            double wordCount = 0;

            while (reader.hasNextLine()) { // processes one line at a time 
                String[] currLine = reader.nextLine().split(" ");
                wordCount += currLine.length + 2; // the start and end tags are not yet added here; hence add 2
                ArrayList<String> sentence = new ArrayList<>();
                sentence.addAll(Arrays.asList(currLine));
                logSum += logProb(sentence);
            }
            reader.close();

            return Math.pow(10, - logSum/wordCount);
            }

        // handle any file not found exceptions
        catch (FileNotFoundException e) {
            System.out.println("File not found.");
            e.printStackTrace();
        }
        return 0.0;
    }

    public double getBigramProb(String first, String second) {
        // transforms unseen words into <UNK>
        if (!unigrams.containsKey(first)){
            first = "<UNK>";
        }
        if (!unigrams.containsKey(second)){
            second = "<UNK>";
        }

        // if our bigram is in our probabilities, return it
        if (bigrams.containsKey(first) && bigrams.get(first).containsKey(second)) {
            return bigrams.get(first).get(second);
        } else {
            // otherwise, calculate the backed-off probability
            return alphas.get(first) * unigrams.get(second); 
            }
    }

    /**
	 * Returns the unigram table of probabilities.
	 */
    public String getUnigramTable() {
        return unigrams.toString();
    }

    /**
	 * Returns the bigram table of probabilities.
	 */
    public String getBigramTable() {
        return bigrams.toString();
    }

    /**
	 * Returns the table of alpha values.
	 */
    public String getAlphaTable() {
        return alphas.toString();
    }


    ////////////////////
    /* HELPER METHODS */
    ////////////////////

    /**
	 * Given two words, increments the bigram count. Used for training.
	 * 
	 * @param first The first word.
     * @param second The following word.
	 */
    private void bigramAdd(String first, String last) {
        // add the first key if it does not exist
        if (!bigrams.containsKey(first))
            bigrams.put(first, new HashMap<String, Double>());

        if (bigrams.get(first).containsKey(last)) {
            // if the bigram already exists, increment by one
            double currCount = bigrams.get(first).get(last);
            bigrams.get(first).put(last, currCount + 1);
        } else {
            // else, add the bigram with a count of 1
            bigrams.get(first).put(last, 1.0);
        }           
    }

    /**
	 * Given a list of words representing a sentence, adds start and end tags, 
     * and replaces words not in vocab with <UNK>
	 * 
	 * @param sent An arraylist of words representing a sentence.
     * @return An arraylist of word representing a formatted sentence.
	 */
    public ArrayList<String> formatSentence(ArrayList<String> sent) {
        ArrayList<String> sentence = new ArrayList<>();
        sentence.addAll(sent);
        for (int i = 0; i < sentence.size(); i++) {
            if (!unigrams.containsKey(sentence.get(i))) {
                sentence.set(i, "<UNK>");
            }
        }
        sentence.add(0, "<s>");
        sentence.add("</s>");
        return sentence;
    }
}