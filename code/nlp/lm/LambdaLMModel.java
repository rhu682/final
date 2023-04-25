package nlp.lm;
import java.util.*;
import java.io.*;

/**
 * A language learning model with lambda smoothing that supports unigram, bigram, and trigram-based calculations.
 */
public class LambdaLMModel {

    ////////////////////////
    /* INSTANCE VARIABLES */
    ////////////////////////
    // The unigram probabilities.
    private HashMap<String, Double> unigrams = new HashMap<String, Double>();

    // The bigram probabilities. Keyed by the first word, then the second. 
    // In other words, for P(Y|X), X is the first key, and Y is the second.
    private HashMap<String, HashMap<String, Double>> bigrams = new HashMap<String, HashMap<String, Double>>();

    // The trigram probabilities. Keyed by the first word, then the second, then the third.
    // In other words, for P(Z|XY), X is the first key, Y is the second, and Z is the third.
    private HashMap<String, HashMap<String, HashMap<String, Double>>> trigrams = 
        new HashMap<String, HashMap<String, HashMap<String, Double>>>();

    // Vocabulary to use.
    private HashSet<String> vocabulary = new HashSet<String>();

    private double lambda;

    ////////////////////
    /* PUBLIC METHODS */
    ////////////////////

    /**
     * Constructs a model trained on a dataset with a given vocabulary.
     * @param filename
     * @param vocab
     * @param lambda
     */
    public LambdaLMModel(String filename, String vocabFile, double lambda){
        
        this.lambda = lambda;
        int wordCount = 0; // used for unigram
        readVocab(vocabFile);

        // adds values to vocab that are (probably) guaranteed to show up
        unigrams.put("<UNK>", 0.0);
        unigrams.put("<s>", 0.0);
        unigrams.put("</s>", 0.0);
        bigrams.put("<UNK>", new HashMap<String, Double>());
        bigrams.put("<s>", new HashMap<String, Double>());

        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-16"));
            String currlineString;
            // Reads through training data, aggregating counts
            while ((currlineString = reader.readLine()) != null) {
                // Adds start and end symbols to the line, and then splits it into an array
                String[] currLine = String.join(" ", "<s>", currlineString, "</s>").split(" ");
                wordCount += currLine.length;

                for (int i = 0; i < currLine.length; i++) {
                    // counts unigrams
                    String currWord = currLine[i];
                    if (!vocabulary.contains(currWord)) {
                        currWord = "<UNK>";
                    }
                    if (unigrams.containsKey(currWord)) {
                        unigrams.put(currWord, unigrams.get(currWord) + 1);
                    } else {
                        unigrams.put(currWord, 1.0);
                    }
                    
                    // count bigrams
                    if (i < currLine.length - 1) {
                        bigramAdd(currLine[i], currLine[i+1]); //bigramAdd handles <UNK> for us. 
                    }

                    // count trigrams
                    if (i < currLine.length - 2) {
                        trigramAdd(currLine[i], currLine[i+1], currLine[i+2]); //trigramAdd handles <UNK> for us. 
                    }
                }
            }
            reader.close();
        } catch (Exception e) {
            System.out.println("Issue in training.");
            e.printStackTrace();
        }
        
        // transforms trigram counts into probabilties
        for (String first : trigrams.keySet()) {
            for (String second: trigrams.get(first).keySet()) {
                double total = bigrams.get(first).get(second);

                for (String third : trigrams.get(first).get(second).keySet()) {
                    double numer = trigrams.get(first).get(second).get(third) + lambda;
                    double denom = total + lambda * vocabulary.size();
                    trigrams.get(first).get(second).put(third, numer/denom);
                }
            }
        }
        
        // transforms bigram counts into probabilties
        for (String first : bigrams.keySet()) {
            double total = unigrams.get(first);
            for (String second : bigrams.get(first).keySet()) {
                double numer = bigrams.get(first).get(second) + lambda;
                double denom = total + lambda * vocabulary.size();
                bigrams.get(first).put(second, numer/denom);
            }
        }

        // transforms unigram counts into probabilities
        for (String key : unigrams.keySet()) {
            unigrams.put(key, unigrams.get(key) / wordCount);
        }        
    }
    
    /**
	 * Given a sentence, return the log of the probability of the sentence based on the LM.
	 * 
	 * @param sentWords the words in the sentence.  sentWords should NOT contain <s> or </s>.
	 * @return the log probability
	 */
    public double logProb(ArrayList<String> sentWords, int gram) {
        if (gram != 1 && gram != 2 && gram != 3) {
            throw new IllegalArgumentException("logProb takes 1, 2, or 3.");
        }

        ArrayList<String> processedSentence = formatSentence(sentWords);
        double logSum = 0;
        for (int i = 0; i < processedSentence.size(); ++i) {

            if (gram == 1) {
                logSum += Math.log10(getProb(processedSentence.get(i)));
            } else if (gram == 2 && i < processedSentence.size() - 1) {
                logSum += Math.log10(getProb(processedSentence.get(i), processedSentence.get(i+1)));
            } else if (gram == 3 && i < processedSentence.size() - 2) {
                logSum += Math.log10(getProb(processedSentence.get(i), 
                                            processedSentence.get(i+1), processedSentence.get(i+2)));
            }
            
        }
        return logSum;
    }

    /**
	 * Given a text file, calculate the perplexity of the text file, that is the negative average per word log
	 * probability
	 * 
	 * @param filename a text file.  The file will contain sentences WITHOUT <s> or </s>.
	 * @return the perplexity of the text in file based on the LM
	 */
    public double getPerplexity(String filename, int gram) {
        if (gram != 1 && gram != 2 && gram != 3) {
            throw new IllegalArgumentException("getPerplexity takes 1, 2, or 3.");
        }

        try {
            File textfile = new File(filename);
            Scanner reader = new Scanner(textfile);
            double logSum = 0;
            double wordCount = 0;

            while (reader.hasNextLine()) { // processes one line at a time 
                String[] currLine = reader.nextLine().split(" ");
                wordCount += currLine.length + 3 - gram;
                ArrayList<String> sentence = new ArrayList<>();
                sentence.addAll(Arrays.asList(currLine));
                logSum += logProb(sentence, gram);
            }
            reader.close();

            return Math.pow(10, - logSum/wordCount);
            }

        // handle any file not found exceptions
        catch (FileNotFoundException e) {
            System.out.println("File not found.");
            e.printStackTrace();
        }

        return 0;
    }

    /**
	 * Returns the unigram probability p(first)
	 * 
	 * @param first
	 * @return the probability of the word occuring
	 */
    public double getProb(String first) {
        // if our unigram is in our probabilities, return it. otherwise, calculate on the fly
        if (unigrams.containsKey(first)) {
            return unigrams.get(first);
        } else {
            return lambda / (lambda * vocabulary.size());
            }
    }

    /**
	 * Returns the bigram probability p(second | first)
	 * 
	 * @param first
	 * @param second
	 * @return the probability of the second word given the first word
	 */
    public double getProb(String first, String second) {
        // if our bigram is in our probabilities, return it. otherwise, calculate on the fly
        if (bigrams.containsKey(first) && bigrams.get(first).containsKey(second)) {
            return bigrams.get(first).get(second);
        } else {
            return lambda / (unigrams.get(first) + lambda * vocabulary.size());
        }
    }

    /**
	 * Returns the trigram probability p(third | first second)
	 * 
	 * @param first
	 * @param second
     * @param third
	 * @return the probability of the third word given the first and second word
	 */
    public double getProb(String first, String second, String third) {
        // if our trigram is in our probabilities, return it. otherwise, calculate on the fly
        if (trigrams.containsKey(first) && trigrams.get(first).containsKey(second) 
            && trigrams.get(first).get(second).containsKey(third)) {
            return trigrams.get(first).get(second).get(third);
        } else if (bigrams.containsKey(first) && bigrams.get(first).containsKey(second)) {
            return lambda / (bigrams.get(first).get(second) + lambda * vocabulary.size());
        } else {
            return lambda / (lambda * vocabulary.size());
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
	 * Returns the trigram table of probabilities.
	 */
    public String getTrigramTable() {
        return trigrams.toString();
    }

    /** Given a file, generates a vocabulary list, then writes it to another file.
     * The vocabulary is also stored, so there's no need to call readVocab() on the same object.
     * All words that appear less than a given threshold will not be included.
     * @param toRead Filepath of file to read.
     * @param toWrite Filepath of file to write. Will overwrite if it already exists.
     * @param threshold How many times a word must appear to be included in the vocabulary.
     */
    public void generateVocab(String toRead, String toWrite, int threshold) {
        // Every word mapped with the number of times it has appeared.
        HashMap<String, Integer> vocab = new HashMap<String, Integer>();

        // Read file, adding counts to words.
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(toRead), "UTF-16"));

            String currline;
            while ((currline = reader.readLine()) != null) {
                String[] words = currline.split(" ");

                // Add each word to our vocab map.
                for (String word : words) {
                    if (vocab.containsKey(word)) {
                        vocab.replace(word, vocab.get(word) + 1);
                    } else {
                        vocab.put(word, 1);
                    }
                }
            }
            reader.close();
        } catch (Exception e) {
            System.out.printf("Error in reading file %s\n", toRead);
            e.printStackTrace();
        }

        // Generate vocab list, removing words not seen frequently enough
        ArrayList<String> vocabToWrite = new ArrayList<String>();
        for (String word : vocab.keySet()) {
            if (vocab.get(word) >= threshold) {
                vocabToWrite.add(word);
            }
        }

        // Add vocab to this object's hashset
        vocabulary.addAll(vocabToWrite);

        // write vocab to file
        try {
            FileWriter myWriter = new FileWriter(toWrite);
            myWriter.write(String.join(" ", vocabToWrite));
            myWriter.close();
        } catch (IOException e) {
            System.out.printf("Error in writing to file %s\n", toWrite);
            e.printStackTrace();
        }
    }

    public void generateVocab(String toRead, String toWrite) {
        generateVocab(toRead, toWrite, 2);
    }

    /** Reads in a vocabulary list.
     * 
     * @param toRead
     */
    public void readVocab(String toRead) {
        vocabulary.add("<s>");
        vocabulary.add("</s>");
        vocabulary.add("<UNK>");

        try {
            FileReader fr=new FileReader(toRead);    
            BufferedReader reader =new BufferedReader(fr);    

            String currline;
            while ((currline = reader.readLine()) != null) {
                String[] words = currline.split(" ");

                // Add each word to our vocab map.
                for (String word : words) {
                    vocabulary.add(word);
                }
            }
            reader.close();
        } catch (Exception e) {
            System.out.printf("Error in reading file %s\n", toRead);
            e.printStackTrace();
        }
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
        // Transforms to <UNK> if necessary.
        if (!vocabulary.contains(first)) {
            first = "<UNK>";
        }
        if (!vocabulary.contains(last)) {
            last = "<UNK>";
        }

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
	 * Given three words, increments the trigram count. Used for training.
	 * 
	 * @param first The first word.
     * @param second The second word.
     * @param third The third word.
	 */
    private void trigramAdd(String first, String second, String third) {
        // Transforms to <UNK> if necessary.
        if (!vocabulary.contains(first)) {
            first = "<UNK>";
        }
        if (!vocabulary.contains(second)) {
            second = "<UNK>";
        }
        if (!vocabulary.contains(third)) {
            third = "<UNK>";
        }

        // add the first key if it does not exist
        if (!trigrams.containsKey(first)) {
            trigrams.put(first, new HashMap<String, HashMap<String, Double>>());
        }

        // add the second key if it does not exist
        if (!trigrams.get(first).containsKey(second)) {
            trigrams.get(first).put(second, new HashMap<String,Double>());
        }

        if (trigrams.get(first).get(second).containsKey(third)) {
            // if the trigram already exists, increment by one
            double currCount = trigrams.get(first).get(second).get(third);
            trigrams.get(first).get(second).put(third, currCount + 1.0);
        } else {
            // else, add the trigram with a count of 1
            trigrams.get(first).get(second).put(third, 1.0);
        }           
    }

    /**
	 * Given a list of words representing a sentence, adds start and end tags, 
     * and replaces words not in vocab with <UNK>
	 * 
	 * @param sent An arraylist of words representing a sentence.
     * @return An arraylist of word representing a formatted sentence.
	 */
    private ArrayList<String> formatSentence(ArrayList<String> sent) {
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
