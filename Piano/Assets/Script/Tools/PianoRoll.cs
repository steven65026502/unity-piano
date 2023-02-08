using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEditor.Experimental.GraphView;
using UnityEngine;

public class PianoRoll
{
    public List<int[]> onset_events;
    public List<int> pedal_events;

    const int clearhead = 9, maxpower = 128;

    public PianoRoll() 
    {
        onset_events = new List<int[]>();
        pedal_events = new List<int>();
    }

    public void Clear()
    {
        if(onset_events != null) onset_events.Clear();
        if(pedal_events != null) pedal_events.Clear();
    }

    static public float PowerToVolume(int power)
    {
        return (float)power / (float)maxpower;
    }
    static public int VolumeToPower(float volume)
    {
        return (int)(volume * maxpower);
    }

    static public int[] NoteToOnsetEvent(Script_Note script_Note)
    {
        Script_NoteHub script_NoteHub = script_Note.transform.parent.GetComponent<Script_NoteHub>();
        int level = script_NoteHub.NoteLevel * Script_NoteHub.NoteLength + Array.IndexOf(script_NoteHub.Notes, script_Note) - clearhead;
        return new int[] { (int)(script_Note.StartTime * 8), level, VolumeToPower(script_Note.Note.volume), (int)(script_Note.EndTime * 8) };
    }

    static public Script_Note OnsetEventToNote(Script_NoteHub[] script_NoteHubs, int[] onset_event)
    {
        int NoteLevel = (onset_event[(byte)PianoRollOnsetEventIndex.Level] + clearhead) / Script_NoteHub.NoteLength;
        int NoteIndex = (onset_event[(byte)PianoRollOnsetEventIndex.Level] + clearhead) % Script_NoteHub.NoteLength;
        Script_NoteHub hub = script_NoteHubs.Where((hub) => hub.NoteLevel == NoteLevel).First();
        return hub.Notes[NoteIndex];
    }
}

public enum PianoRollOnsetEventIndex
{
    StartTime = 0,
    Level,
    Power,
    EndTime
}
