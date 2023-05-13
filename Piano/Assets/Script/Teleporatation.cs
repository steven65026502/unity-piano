using UnityEngine;
using TMPro;
using UnityEngine.UI;
using Newtonsoft.Json;
using System.Threading.Tasks;

[System.Serializable]
public class StartMessage
{
    public string signal;
    public float temperature;
    public float p;
}

public class Teleporatation : MonoBehaviour
{
    public TextMeshProUGUI buttonText;
    public Slider temperatureSlider;
    public Slider pSlider;
    public Slider minLengthSlider;

    private float temperatureValue;
    private float pValue;
    private float minLengthValue;
    public bool isModel1Selected = true;
    public model modelScript;

    private void Start()
    {
        temperatureSlider.onValueChanged.AddListener(OnTemperatureSliderChanged);
        pSlider.onValueChanged.AddListener(OnPSliderChanged);
        minLengthSlider.onValueChanged.AddListener(OnMinLengthSliderChanged);

        modelScript = GameObject.Find("model").GetComponent<model>();
    }

    public void OnTemperatureSliderChanged(float value)
    {
        temperatureValue = value;
    }

    public void OnPSliderChanged(float value)
    {
        pValue = value;
    }

    public void OnMinLengthSliderChanged(float value)
    {
        minLengthValue = value;
    }

    public async void OnClick()
    {
        if (temperatureValue < 0.5f)
        {
            temperatureValue = 0.5f;
            temperatureSlider.value = temperatureValue;
        }
        if (pValue < 0.5f)
        {
            pValue = 0.5f;
            pSlider.value = pValue;
        }
        if (minLengthValue < 1000)
        {
            minLengthValue = 1000;
            minLengthSlider.value = minLengthValue;
        }

        string selectedModel = modelScript.currentModel;
        await Manager.Instance.UpdateAndSendMessageToServer(temperatureValue, pValue, minLengthValue, selectedModel);

        // 這裡寫下 teleportation 按鈕被點擊後的行為
        // 當按鈕被點擊時，執行以下操作：

        // 隱藏指定的音符
        foreach (Script_NoteHub noteHub in FindObjectsOfType<Script_NoteHub>())
        {
            for (int i = 0; i < Script_NoteHub.NoteLength; i++)
            {
                noteHub.HideNote[i] = false;
            }
        }
        for (int i = 0; i < 9; i++)
        {
            Script_NoteHub noteHub0 = GameObject.Find("NoteHub0").GetComponent<Script_NoteHub>();
            noteHub0.HideNote[i] = true;
        }
        for (int i = 11; i > 1; i--)
        {
            Script_NoteHub noteHub8 = GameObject.Find("NoteHub8").GetComponent<Script_NoteHub>();
            noteHub8.HideNote[i] = true;
        }

        // 暫停時間軸
        Manager.Instance.TimeStop();

        // 將時間和音符數據轉換為 JSON 格式，並輸出到控制台
        string jsonData = JsonConvert.SerializeObject(Manager.Instance.timeAndNotes);
        Debug.Log(jsonData);

        // 清空時間和音符數據
        Manager.Instance.timeAndNotes.Clear();
    }
}